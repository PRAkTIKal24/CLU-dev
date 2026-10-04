# figure-layout-fix — results-analyst report

Task + acceptance criterion: re-lay-out the three figures the render pass flagged as illegible at their printed box (V5 Fig 1 / Fig 2 / Fig C.2), restore legends + every in-figure numeric label, execute the main-text/appendix split of V5 Fig 1, check V2's three figures at the de-scoped build's widths, install everywhere, rebuild, report every page split.
Status: **done, with one flagged and unfixed change** — the three cramped figures now meet the type targets with all legends and all 12 numeric labels restored; **all five re-renders verified IDENTICAL DATA** against the original generators' banked tap digests; V5 main text **held at 4.00 pp**; **V5 total went 9 pp → 10 pp**, and I prove below that the *main* figure is not the cause (a build with only the main-figure change is byte-for-byte the same 9 pp / 4.00 pp / END 8.72 pp as the control). V2 reframe **13 pp / main 5.67 pp unchanged**; V2 de-scoped **14 pp / main 6.14 pp unchanged**.

**⚠ DOWNSTREAM RECONCILIATION LIST — THIS REPORT CONTAINS ONE, IT NEEDS AN OWNER (§8).** The consolidated caption list has **5 live items** (down from the render pass's 9: four were made moot by the restored legends) **+ 3 new items** created by this pass (a new appendix figure caption I had to write, and the V5 figure renumbering 2→3, 3→4). ⛔ I edited **no existing caption**.

**⚠ SECOND FLAG (§6): V5 is now 10 pp.** Measured trade table: with the main figure fixed, **either** Fig 2 **or** Fig C.2 can be made legible inside 9 pp — **not both** (Fig C.2 alone lands END at exactly 9.00 pp with **0.45 pt** of the text block left). Nine pages and two legible appendix figures are mutually exclusive on this source. Advisor's call; the 9-pp fallback recipe is priced in §6.3.

**DIAL DECLARATION (echoed): none — figure regeneration from banked artifacts. Zero new measurements. Laundering control: n/a. Falsifies: n/a. Does NOT falsify: n/a.** (Pre-registration rule: not triggered — the acceptance criterion is a layout/typography outcome, not a measured ratio, exponent, slope or law.)

---

## 1. What I did

1. Re-measured every printed image box from the built PDFs (`mutool draw -F trace`), including the **de-scoped** build, which the render pass never measured (§2).
2. **Measured the main-text height budget empirically** instead of assuming it: a placeholder sweep over the Fig-1 box height at three widths (§3). The budget is **1.42 in**; 1.44 in spills to 5 pp. It is width-independent.
3. Re-laid-out five figures (§4) at their new printed widths, with legends restored and every deleted numeric label back; ran each under the **previous pass's data tap** and compared to the *original generators'* banked digests (§5).
4. Installed into all three folders, rebuilt each ×3, and decomposed the page cost figure-by-figure with five diagnostic builds (§6).
5. Swept identifying strings and internal labels (§7); verified `papers/v2-short/**` + `papers/v5-short/**` byte-untouched (46-file manifest diff empty) and the two V2 folders' file lists unchanged.

### 1.1 Environment / provenance
Repo HEAD **`7fcef501fb1aeae33a8d149b046c9b6126dcecd7`**, `git status --porcelain` empty (all artifacts are under `.claude/`, gitignored). Main venv: **matplotlib 3.10.8, numpy 2.4.1, Pillow 12.1.0**; TeX Live 2026 / pdfTeX; `mutool`, `pdftotext` (poppler). No JAX needed. Build = `pdflatex -interaction=nonstopmode submission.tex` ×3.

### 1.2 Flag-provenance table (inherited — this pass made ZERO measurements)
Every number plotted here comes from the same banked artifacts the render pass documented; the tap proves the arrays are unchanged (§5), so the render pass's provenance table transfers verbatim. Restated for the two figures whose *scripts* are new in this pass:

| figure | source artifact(s) | seeds | non-default flags in effect at measurement time |
|---|---|---|---|
| V5 Fig 1 (main) **and** the new V5 Fig 2 (appendix companion) | `outputs/t-lever-forgetting/s4b_jacobian.json` (115 rows) + `outputs/v5-gate/e1c_vcurve.json` | designed 42–46 (5) + emergent 42–44 (3) | one-step Jacobian, dim 4, hidden 64, ε=0.05, `langevin_noise="fdt"`, Newtonian kinetic mode, laptop-CPU float64; colorbar floor marker = ring-profile probe floor 1.7×10⁻¹² |
| V5 Fig 3 (two instruments) | `outputs/v5-vcurve-validation/m1_emergent.json` | emergent 42–44 (3) | 48 γ ∈ geomspace(0.002,0.5), δ∈{0.05,0.2,0.5} (plotted δ=0.05), dt=0.05, **T=0 deterministic**, retie=True, n_chunks 8000, min_steps 20000, n_halflives 30, ov_min 0.3 |
| V5 Fig 4 (vault) | `outputs/v5-vcurve-validation/m3_{emergent_VDS,designed_crosscheck,emergent_X,fpt_emergent,fpt_samesite}.json` | emergent 42–44 (3), designed cross-check 42–44 | dt=0.05, γ=0.05, γ_φ∈{0,0.1,0.2,0.3,0.5}, Δ=0.5 rad, **noise="fdt"**, retie=True, T∈{4,8}×10⁻³, gamma_max=0.9, gate="compact", width=0.25, uniform_radius=50.0 |
| V2 Fig 3 (GMOR) | `outputs/f1-gmor-condensate/{gmor_condensate,angular_tilt_contrast}.npz` | designed 42–46 + anchored3000 42–44 (8 checkpoints) | probe-only, no retraining; linear ambient spurion δ∈[10⁻⁸,0.3] (10 pts); float64 |

**Seed neutralisation mapping (unchanged from the render pass):** file order, **42→1, 43→2, 44→3** (and 45→4, 46→5 for the 5-seed designed families).

---

## 2. Printed boxes, before this pass — including the de-scoped build the render pass never measured

| variant | figure | `\includegraphics` width | printed box (measured) |
|---|---|---|---|
| V5 PALM | Fig 1 damping optimum | `0.60\linewidth` | 237.616 × **80.283** pt = 3.300 × **1.115** in |
| V5 PALM | Fig 2 two instruments (App B) | `0.74\linewidth` | 293.057 × **82.056** pt = 4.070 × **1.140** in |
| V5 PALM | Fig 3 vault "C.2" (App C) | `0.80\linewidth` | 316.808 × **70.031** pt = 4.400 × **0.973** in |
| V2 reframe | Fig 1 Mo (main) | `0.68\linewidth` | 269.301 × 147.299 pt = 3.740 × 2.046 in |
| V2 reframe | Fig 2 anchor cure (App) | `0.86\linewidth` | 340.571 × 130.036 pt = 4.730 × 1.806 in |
| V2 reframe | Fig 3 GMOR (App) | `0.60\textwidth` | 237.609 × **180.255** pt = 3.300 × **2.504** in |
| **V2 de-scoped** | Fig 1 / Fig 2 / Fig 3 | same three widths | **269.283 × 147.289 / 340.568 × 130.035 / 237.598 × 180.247 pt** |

**Answer to the "check V2 at the de-scoped build's widths" instruction: the de-scoped boxes are the same boxes** (`\textwidth = \linewidth = 5.5 in` in both, same `.sty`); the three agree with the reframe to **≤0.011 pt**. So the render pass's V2 verdicts transfer: **F1 comfortable, F2 met**, **F3 marginal** — its in-axes notes printed at **7–7.5 pt** (below the ≥8 pt target) and all four legends had been deleted. **V2 F3 therefore fails the rule and is fixed here; V2 F1 and F2 pass and are untouched.**

Note also: the de-scoped folder was still shipping the **pre-re-render** PNGs (`1529629f… / 887980781e… / b988a66c…`), i.e. the one-step follow-up its own BUILD-NOTE §10 owed. That is discharged here (§6.4).

---

## 3. The main-text height budget, measured (this replaces an assumption in the V5 BUILD-NOTE)

`\includegraphics` boxes were replaced with flat placeholder PNGs of an exact size, built ×3, and the References marker located. Only the *box* matters to LaTeX, so a placeholder is an exact instrument.

| width | box height | pages | main text | verdict |
|---|---|---|---|---|
| `0.60\linewidth` | 1.115 in (the shipped box) | 9 | 4.00 pp | fits |
| `0.60` | 1.20 / 1.25 / 1.30 / 1.35 / 1.40 in | 9 | 4.00 pp | **fits** |
| `0.60` | **1.42 in** | 9 | 4.00 pp | **fits — the ceiling** |
| `0.60` | 1.44 / 1.46 / 1.48 / 1.50 in | **10** | **5.00 pp** | spills |
| `0.70` | 1.302 in (banked aspect at 0.70) | 9 | 4.00 pp | fits |
| `0.70` | 1.45 in | 10 | 5.00 pp | spills |
| `0.45` | 1.45 in | 10 | 5.00 pp | spills |

**Two findings.**
1. **The binding constraint is height and nothing else** — 1.45 in spills at 0.45, 0.60 *and* 0.70 `\linewidth`. The task's diagnosis ("the cause is aspect ratio, not width") is confirmed as a *measurement*, not an inference.
2. **There is real headroom: 1.42 in vs the shipped 1.115 in, i.e. +0.305 in (+22.0 pt, +27 %).** The main text's last body line sits at y=702.16 pt against a text block bottom of 720 pt, i.e. **17.84 pt of slack**, and float repacking converts that into the 22 pt of figure height.
3. ⚠ **This contradicts `papers/palm-variant/v5/BUILD-NOTE.md` §6**, which states that widths *0.62, 0.64, 0.66, 0.70, 0.76 and 0.84 "were each built and all spill onto p. 5 (and take the total to 10 pp)"*. On the source **as it stands today**, `0.70\linewidth` at the banked aspect (box 3.850 × 1.302 in) builds **9 pp with main text at 4.00 pp**. Either the prose moved after that note was written, or the note's builds used a different figure aspect. I report what I measured; the BUILD-NOTE line should be re-checked or annotated by whoever owns it.

---

## 4. What was re-laid-out, and why each choice

Every new canvas is rendered **at its printed width** (canvas width = printed width, verified: e.g. 229.691 pt / 72 = 3.1901 in against a 3.190 in canvas), so **the specified pt sizes are the printed pt sizes, scale 1.000×**.

### 4.1 V5 Fig 1 — the main-text figure. **The prescribed split is vacuous as written; here is what I did instead, and why.**
The task's rule 2 assumes "the complete **multi-panel** version". **V5 Fig 1 is a single panel plus a colorbar** (`scratch/v5-revision-1/fig1_collapse.py`: one `plt.subplots(figsize=(10.4,3.5))`, one axes). There are no siblings to separate, so "money panel stays, the rest goes to the appendix" has no referent. Per rule 3 I am saying so rather than splitting it badly, and I executed the *analogue* that is well defined:

- **Main text: reduced.** Width `0.60\linewidth` → **`0.58\linewidth`** (3.300 → **3.190 in**, −3.3 %), height **1.400 in** (inside the measured 1.42 in ceiling, 0.02 in of margin). Aspect **2.96 → 2.28**. **Legend restored** (2 entries, 8 pt, upper-centre, in the empty interior of the "V"); slope labels at 8 pt; colorbar retained but slimmed; the render pass's **6.5 pt** colour-key note is **gone** (nothing below 8 pt survives anywhere).
- **Appendix: the complete version at full size, under a new figure number** (`figA1_damping_optimum_full.png`, printed **4.950 × 2.900 in** at `0.90\linewidth`, inserted at the head of Appendix B). It carries everything the main box cannot: the two-family legend with the evidence tiers, the `γ=γ_crit=2εμ` marker, the **μ→0 flat-coset callout**, and the **crimson probe-floor annotation on the μ² colourbar (1.7×10⁻¹²)**.
- **Why not "square-ish":** measured. At the 1.42 in ceiling a square figure is 1.42 × 1.42 in, whose axes would be ≈1.0 × 0.9 in — 3.3 decades of x in one inch, *worse* than what ships. I rendered the intermediate case (`0.46\linewidth`, 2.530 × 1.400 in, aspect 1.81) and it fails visibly: the y-label and colorbar label clip and the 2-entry legend fills the panel. Sheet: `.claude/scratch/figure-layout-fix/render46/fig1_damping_optimum.png`. **Square-ish and legible are not simultaneously reachable under a 1.42 in ceiling; I chose legible.**

### 4.2 V5 Fig 2 (two instruments, App B) → **1×3, wider and taller**
`0.74\linewidth` × 1.140 in → **`0.90\linewidth` = 4.950 × 2.250 in**. Legends restored as three per-panel legends that decode *every* mark without the caption: (a) instrument key — Jacobian / rollout rate / rollout threshold; (b) decay-rate ratio / threshold ratio; (c) seed 1/2/3 colour key. All at 8 pt. Internal IDs (`I-J/I-R1/I-R3`, `Γ_jac/Γ_R3`, `n_jac/n_R1`) stay stripped.

### 4.3 V5 Fig C.2 (vault, App C) → **1×4 becomes 2×2**
`0.80\linewidth` × 0.973 in → **`0.86\linewidth` = 4.730 × 3.550 in**, 2×2 with a shared seed-colour legend on a bottom row. **All four panel legends restored** and **all twelve deleted numeric labels restored**: the nine hop fractions in (c) — 5.5 / 0.7 / 0.0, 43.0 / 10.2 / 0.0, 2.4 / 0.3 / 0.0 % — and the three vault factors in (d) — **>1379× / 35× / >1290×**. Panel titles are now descriptive ("(a) refrigerator law", "(b) γ_eff⁻² diffusion law", "(c) confinement", "(d) same-site first passage") instead of bare letters. Q-labels stay stripped.

### 4.4 V2 Fig 3 (GMOR, App) → taller, at both V2 variants
`0.60\textwidth` × 2.504 in → **`0.80\textwidth` = 4.400 × 3.600 in**. Four legends restored, all at 8 pt, including the **panel-(c) "max dev 2e−15" number** that had been deleted, and the panel-(a) marker key ("anchored, 3000 ep" open circles / "designed, 150 ep" filled squares) — which **makes render-pass caption item 8 moot**. Checkpoint short-tags stay stripped.

### 4.5 The measurement that shows this is a real fix — axes box, before → after

| figure | data panels | axes height (in) | axes width (in) |
|---|---|---|---|
| V5 Fig 1 (main) | 1 | **0.619 → 0.903** (+46 %) | 2.161 → 2.233 |
| V5 Fig 2 → Fig 3 (two instruments) | 3 | **0.730 → 1.829** (+151 %) | 0.927 → 1.241 (+34 %) |
| V5 Fig C.2 → Fig 4 (vault) | 4 | **0.582 → 1.189** (+104 %) | 0.680 → **1.900** (+179 %) |
| V2 Fig 3 (GMOR) | 4 | **0.792 → 1.313** (+66 %) | 1.175 → 1.698 (+45 %) |
| V5 Fig 2 (new appendix companion) | 1 | — → 1.740 | — → 3.143 |

### 4.6 Type targets — met, with no sub-8 pt text anywhere

| figure | ticks (≥7) | axis labels (≥8) | titles (≥9) | legend (≥8) | in-figure numeric labels (≥8) |
|---|---|---|---|---|---|
| V5 Fig 1 (main) | **7.0** | **8.0** | **9.0** | **8.0** (2 entries) | **8.0** (slope ±1) |
| V5 Fig 2 (new, App B) | **8.0** | **9.0** | **9.5** | **8.5** (2 entries) | **8.5** (incl. 1.7×10⁻¹² floor note) |
| V5 Fig 3 (two instruments) | **7.0** | **8.0** | **9.0** | **8.0** (3+2+3 entries) | — (none in this figure) |
| V5 Fig 4 (vault) | **7.0** | **8.0** | **9.0** | **8.0** (4+3+3+3+3 entries) | **8.0** (all 12 restored) |
| V2 Fig 3 (GMOR) | **7.0** | **8.0** | **9.0** | **8.0** (3+3+1+1 entries) | **8.0** (2.2×10⁻¹⁶, 2e−15 in legends) |

Machine check: `grep fontsize=` over the five new scripts, excluding tick settings ⇒ **0 occurrences below 8.0**. (The render pass's shipped V5 Fig 1 carried a 6.5 pt note and V5 Fig C.2/Fig 2 carried 7 pt notes; those are gone.)

Print-size before/after sheets (300 dpi, rendered at the actual printed width — this is what a reviewer sees):
`.claude/outputs/figure-layout-fix/compare_{v5fig1,v5fig2,v5figC2,v2fig3}.png`

---

## 5. Verification that no plotted value moved

Instrument: the previous pass's `tap.py` + `compare.py`, unmodified, run via `.claude/outputs/figure-layout-fix/run_tap.py` (savefig redirected so nothing could touch a banked artifact). Comparison is against the **original generators'** banked digests in `.claude/outputs/figure-render-pass/digests/*_orig.json` — i.e. against `fig1_collapse.py`, `a1_analyse.py`, `a3_fig.py`, `analyze_and_figure.py` as they were before either pass touched anything.

| new render | whole-run sha1 | original whole-run sha1 | non-empty data calls | verdict |
|---|---|---|---|---|
| `v5f1_main.py` | `36876fc327190ab9…` | `36876fc327190ab9…` | 23 vs 23 | **IDENTICAL DATA** (whole-run hash equal) |
| `v5f1_appx.py` | `36876fc327190ab9…` | `36876fc327190ab9…` | 23 vs 23 | **IDENTICAL DATA** (whole-run hash equal) |
| `v5f2_tall.py` | `cb22bfd28223a839…` | `cb22bfd28223a839…` | 46 vs 46 | **IDENTICAL DATA** (whole-run hash equal) |
| `v5fC2_tall.py` | `6c11f9b07d1d64e7…` | `6c11f9b07d1d64e7…` | 38 vs 38 | **IDENTICAL DATA** (whole-run hash equal) |
| `v2f3_tall.py` | `f674c5ad9e462663…` | `f674c5ad9e462663…` | 66 vs 66 | **IDENTICAL DATA** (whole-run hash equal) |

`compare.py` prints `in ORIGINAL not in NEW : 0` / `in NEW not in ORIGINAL : 0` for all five. This is *stronger* than the render pass's result for two of the figures, where the whole-run digest differed by empty legend-proxy calls: here the whole-run hash is bit-equal to the original in every case. Digests: `.claude/outputs/figure-layout-fix/digests/*.json`.

**What did change, and it is not a value:** axis limits. V5 Fig 1's appendix companion uses `ylim (0.8, 1000)` instead of `(0.8, 300)` to make room for the callout; V5 Fig 4's panels (a)/(b)/(c)/(d) use `(0.075, 6.5) / (0.115, 14) / (0.045, 90) / (1e2, 2e10)` instead of `(0.09,1.9) / (0.5,10) / (0.045,3.0) / (1e2,5e6)`, to open the headroom the restored legends and bar labels need; V2 Fig 3 (a)/(b)/(d) similarly. **No data point is clipped by any new limit** — every limit was widened, never narrowed, and the tap confirms the plotted arrays are unchanged.

---

## 6. Rebuilds and page splits — every build, measured

Instrument: `.claude/outputs/figure-layout-fix/measure.py`, which wraps the render pass's `pagesplit.py` (fractional pages from PDF word boxes against the text block, top 72 pt / bottom 720 pt) plus `imgbox.py` (image boxes from `mutool draw -F trace`). "END" is where the last body line falls, in fractional pages — the number that says how much room is left.

### 6.1 V5 PALM variant — the decomposition (5 builds, all from the same byte-identical baseline)

| build | pages | main text | END | Fig-1 box | Fig-2/3/4 boxes |
|---|---|---|---|---|---|
| **control** (as found) | 9 | **4.00 pp** | 8.72 pp (178.45 pt spare) | 3.300 × 1.115 | 4.070 × 1.140 ; 4.400 × 0.973 |
| B: **main Fig 1 change only** | **9** | **4.00 pp** | **8.72 pp (178.45 pt spare)** | 3.190 × 1.400 | unchanged |
| ii: main Fig 1 + Fig 2 tall | 9 | 4.00 pp | 8.88 pp (75.91 pt spare) | 3.190 × 1.400 | 4.950 × 2.250 ; unchanged |
| iii: main Fig 1 + Fig C.2 2×2 | 9 | 4.00 pp | **9.00 pp (0.45 pt spare)** | 3.190 × 1.400 | unchanged ; 4.730 × 3.550 |
| C: main + both appendix figures | **10** | 4.00 pp | 9.35 pp | 3.190 × 1.400 | 4.950 × 2.250 ; 4.730 × 3.550 |
| **D = SHIPPED** (C + the new companion figure) | **10** | **4.00 pp** | 9.62 pp | 3.190 × 1.400 | 4.950 × 2.900 (new) ; 4.950 × 2.250 ; 4.730 × 3.550 |

**Build B settles the ⚠ condition.** The task says *"if the **reduced main figure** changes it, report and stop"*. Build B is the control plus the reduced main figure and nothing else: **9 pages, main text 4.00 pp, END 8.72 pp, last-page slack 178.45 pt — every number identical to the control.** The reduced main figure changes the split by exactly nothing. The stop condition does not fire.

**What does cause the 10th page:** appendix figure height, which rule 1 of the task authorises as "free". Row iii is the sharp fact — **making Fig C.2 legible, on its own, consumes the entire remaining page: END 9.00 pp with 0.45 pt of the text block left.**

Overfull/underfull, control → shipped: overfull **2 → 2** (`91.68 pt`, `406.18 pt`, both pre-existing and unchanged), undefined **0 → 0**, underfull `\vbox`/`\hbox` **1 → 3** (the two new ones are float-column slack from the taller appendix figures, cosmetic).

### 6.2 V2 reframe and V2 de-scoped — no cost at all

| variant | | pages | main text | END | Fig 3 box |
|---|---|---|---|---|---|
| `neurreps-variants/v2` | control | 13 | **5.67 pp** | 12.58 pp | 3.300 × 2.504 in |
| | **shipped** | **13** | **5.67 pp** | 12.70 pp | **4.400 × 3.600 in** |
| `v2-neurreps-descoped` | control | 14 | **6.14 pp** | 13.54 pp | 3.300 × 2.503 in |
| | **shipped** | **14** | **6.14 pp** | 13.66 pp | **4.400 × 3.600 in** |

Overfull 0 → 0 and undefined 0 → 0 in both; underfull 15 → 15 (reframe) and 13 → 13 (de-scoped) — unchanged. Enlarging V2 Fig 3 by 1.10 in costs **0.12 pp** of appendix and **no page**.
*Instrument note:* my main-text reading for the de-scoped build is **6.14 pp** where its BUILD-NOTE records 6.19 pp. That is a marker-choice difference of the same kind the de-scoped BUILD-NOTE itself records (it read the reframe at 5.71 against a recorded 5.69). **The comparison that matters is control → shipped on one instrument, and it is 6.14 → 6.14.**

### 6.3 The 9-pp fallback, priced (if the Advisor wants 9 pp back)
With the main figure and Fig 2 at their new sizes, I swept Fig C.2's box height (4.730 in wide, placeholders):

| Fig C.2 height | pages | END |
|---|---|---|
| 1.60 in | 9 | 8.93 pp |
| 1.90 in | 9 | 8.96 pp |
| **2.20 in** | **9** | **9.00 pp (0.45 pt spare) — the ceiling** |
| 2.50 in | 10 | 9.02 pp (spills) |

**So the 9-pp recipe is: drop the new appendix companion, and cap Fig C.2 at 2.20 in.** At 2.20 in × 4.730 in a 2×2 grid gives rows of 1.10 in ⇒ **axes ≈0.62 in tall**, into which a 3–4-entry 8 pt legend (0.45–0.58 in) does not fit; a 1×4 at that height gives panels 1.18 in wide, into which no 8 pt legend fits at all. **Measured conclusion: within 9 pp, V5 Fig C.2 cannot carry its legends.** The options are (a) 10 pp, (b) drop panels from Fig C.2, (c) leave it illegible. I shipped (a) because rule 1 of the task states appendix height is excluded from the page limit at both venues; (b) is a content decision I am not authorised to make (rule 3).

### 6.4 The de-scoped folder's owed follow-up is discharged
`papers/v2-neurreps-descoped/BUILD-NOTE.md` §10 owed: copy the three regenerated PNGs, apply the caption edits, rebuild ×3. **Done for the PNGs and the rebuild** — `fig1_mo_headtohead.png` → `47d6d459…`, `fig2_anchor_cure_laws.png` → `f25718b9…` (the render pass's versions, identical to the reframe's), `fig3_gmor_condensate.png` → `cb8d0791…` (this pass's). ⛔ **Its §10 also says "Do not copy the PNGs without the caption edits", and my task forbids caption edits.** I followed my task; the caption list in §8 is the discharge, and it must be worked before this build is sent anywhere.

---

## 7. Hygiene and untouched-build checks

- **Identifying strings in the five new PNGs** (`strings -a | grep -Eic 'user|Desktop|CHLU|claude|Forgis|x10719pj|worktree|/Users|agent/|scratch|papers|matplotlib'`): **0, 0, 0, 0, 0.** Positive control on a banked pre-render PNG: **1** (matplotlib's `Software` tEXt chunk). All new PNGs are saved with `metadata={"Software": None}`.
- **Internal labels reaching the canvas** (regex `SF-3|F-1|Q[1235]|T5|T6|Cor-13|CM-16|I-J|I-R|s4[2-6]|a3000|d150|_jac|_R[13]|scratch|.claude|/Users` over `set_title|set_xlabel|set_ylabel|suptitle|set_label|.text(|fig.text|label=` lines): **0 hits in all five new scripts.** Positive control on the original `a3_fig.py`: **4 hits.**
- **`papers/v2-short/**` + `papers/v5-short/**`:** 46-file `shasum` manifest taken before any work and re-taken at the end — **`diff` empty, byte-untouched.** (`.claude/scratch/figure-layout-fix/live_builds_{BEFORE,AFTER}.txt`.)
- **Folder file lists** of `neurreps-variants/v2` and `v2-neurreps-descoped`: **unchanged** — I deleted the `submission.{aux,log,out}` my builds created in those two folders, because neither folder had them before (the `palm-variant/v5` folder did, so its three are left in place). This avoids repeating the incident recorded in the de-scoped BUILD-NOTE §9.
- **⚠ NEW HYGIENE FINDING, outside my scope but visible in the rebuilt PDF:** the render pass stripped internal instrument IDs from the *figures*, but **V5's Tables 2 and 3 still print `I-J / I-R1 / I-R2 / I-R3`, `Γ_jac/Γ_R3` and `n_jac/n_R1` in their column headers** (p. 6 of the shipped build, `submission.tex` around the Table 2/3 blocks). Same class of internal label, same paper, not covered by any existing list. Needs an owner.

---

## 8. CONSOLIDATED CAPTION-EDIT LIST — one list, keyed by file and figure. **I edited none.**

**V5 figure renumbering caused by the new appendix figure** (no `\ref` in the V5 source points at any figure by number, and the only prose occurrence of "Figure 1" is a citation to *Buchbinder & Petrank's* Figure 1, so nothing in the text breaks — but every downstream document that says "V5 Fig 2 / Fig C.2" must be re-keyed):

| was | is now | file, line of `\includegraphics` |
|---|---|---|
| Figure 1 (main, `fig:collapse`) | **Figure 1** (unchanged) | `palm-variant/v5/submission.tex` l.79 |
| — | **Figure 2** (NEW, App B, `fig:collapse-full`) | l.159 |
| Figure 2 (two instruments) | **Figure 3** | l.209 |
| Figure 3 (the vault, a.k.a. "Fig C.2") | **Figure 4** | l.268 |

### `papers/palm-variant/v5/submission.tex`
| # | figure / line | item | status |
|---|---|---|---|
| **1** | **Fig 3** (two instruments), caption **l.210** | Names `(I-J, lines)`, `(I-R3, circles)`, `(I-R1, dotted)`. Those IDs are not in the figure; its legend now reads **"Jacobian" / "rollout rate" / "rollout threshold"**. Re-word to match. | **LIVE — hard defect** (carried from render pass item 1) |
| **2** | Fig 3, caption l.210 | Panel (a)'s y-axis is `n_{1/2}` in **steps**; the axis label already says so. | **optional** (render pass item 2) |
| **3** | **Fig 4** (vault), caption **l.269** | Opens *"panels are labelled by the pre-registered prediction each tests"* — **still false**; the Q-labels are stripped and the panels now carry descriptive titles. Delete the clause. | **LIVE — hard defect** (render pass item 3) |
| ~~4~~ | Fig 4 (a) | ~~add "circles T=4×10⁻³, squares T=8×10⁻³"~~ | **MOOT — the key is back in the figure legend** |
| ~~5~~ | Fig 4 (c) | ~~add the red/grey/blue colour key~~ | **MOOT — legend restored ("no hole (γ=0.05)", "scalar control (γ=0.525)", "γ_φ hole")** |
| ~~6~~ | Fig 4 (c) | ~~drop "and hop fraction" or point at §3.2~~ | **MOOT — all nine hop-fraction labels are back on the bars** |
| ~~7~~ | Fig 4 (d) | ~~add the red/blue key~~ | **MOOT — legend restored ("no hole (same site)", "γ_φ hole", "censored lower bound")** |
| **7b** | Fig 4, caption l.269 | Panel (d) alone is at `T=4×10⁻³`; the caption's provenance parenthetical says `T∈{4,8}×10⁻³`. Still worth stating for (d). | **LIVE — minor** (residue of render pass item 7) |
| **N1** | **Fig 2** (NEW), caption **l.160** | **I wrote this caption** because a new figure cannot ship without one. It is factual and inherits Fig 1's provenance parenthetical verbatim, but it has had **no writer sign-off**. Review it. | **NEW — needs sign-off** |
| **N2** | Fig 1, caption l.80 | The main figure's colourbar still carries the **crimson probe-floor tick** but the 6.5 pt note that decoded it is gone (it was below the type target). Either add *"crimson tick on the colourbar: the flat-coset probe floor, μ²=1.7×10⁻¹²"* to the caption, or point the reader at Figure 2, which annotates it in full. | **NEW — live, one un-decoded mark in the headline figure** |

### `papers/neurreps-variants/v2/submission.tex` **and** `papers/v2-neurreps-descoped/submission.tex` (identical caption text in both)
| # | figure / line | item | status |
|---|---|---|---|
| ~~8~~ | Fig 3 (GMOR), caption **l.340** (reframe) / **l.350** (de-scoped) | ~~add "open circles = anchored 3000-epoch, filled squares = designed 150-epoch"~~ | **MOOT — the marker legend is restored in-figure ("anchored, 3000 ep" / "designed, 150 ep")** |
| **9** | same caption | Panel (c)'s "max dev" is now printed **in the figure** as `2e−15`; the caption's own value for the *absolute-deviation* floor is `≤1.33×10⁻¹⁵` for panel (b). Different quantities, but adjacent — worth a consistency read. Panel (d)'s "slope −1.05" is caption-only, as before. | **LIVE — minor** (render pass item 9, re-scoped) |
| **N3** | both files | Neither caption states the figure's new size/layout; no edit is required by the change itself. Confirmed: **no caption in either V2 file names a stripped seed tag or a stripped short-tag**, so the hygiene posture is caption-safe. | **no action** |

**Also owed, not a caption:** the V5 **Table 2 / Table 3 header IDs** (`I-J / I-R1 / I-R2 / I-R3`, `Γ_jac/Γ_R3`, `n_jac/n_R1`) — see §7.

---

## Git footprint

**None.** `git status --porcelain` empty; HEAD unchanged at `7fcef501fb1aeae33a8d149b046c9b6126dcecd7`. No tracked file created, modified or deleted. Everything written is under `.claude/` (gitignored):
- **Installed** — `papers/palm-variant/v5/figs/{fig1_damping_optimum,figA1_damping_optimum_full,fig2_two_instruments,figC2_vault_emergent}.png` (`85845e80…`, `7fe9b215…`, `728c5082…`, `79e1a776…`) + `submission.tex` (4 layout edits: three `width=` values, one new `figure` environment) + rebuilt `submission.{pdf,aux,log,out}`; `papers/neurreps-variants/v2/figs/fig3_gmor_condensate.png` (`cb8d0791…`) + `submission.tex` (one `width=` edit) + rebuilt `submission.pdf`; `papers/v2-neurreps-descoped/figs/{fig1_mo_headtohead,fig2_anchor_cure_laws,fig3_gmor_condensate}.png` + `submission.tex` (one `width=` edit) + rebuilt `submission.pdf`.
- **Report dir** `.claude/outputs/figure-layout-fix/`: the five render scripts (`v5f1_main.py`, `v5f1_appx.py`, `v5f2_tall.py`, `v5fC2_tall.py`, `v2f3_tall.py`), the free-aspect canvas helper `size2.py`, the tap runner `run_tap.py` + `digests/`, the measurement instruments `measure.py` / `imgbox.py` / `build.sh`, the print-size sheets `compare_*.png`, and `figs_final/` (copies of the five installed PNGs).
- **Scratch** `.claude/scratch/figure-layout-fix/`: `baseline-v5/`, `baseline-v2/`, `baseline-v2ds/` (pre-pass copies of all three folders), `figs_before/`, `pages/` (rasterised built pages), the two live-build manifests and the two folder file lists.

**Exact commands for reproduction** (cwd `/Users/user/Desktop/CHLU`, main venv):
```
FRP_OUT=<dir> .venv/bin/python .claude/outputs/figure-layout-fix/v5f1_main.py     # 3.190 x 1.400 in
FRP_OUT=<dir> .venv/bin/python .claude/outputs/figure-layout-fix/v5f1_appx.py     # 4.950 x 2.900 in
FRP_OUT=<dir> .venv/bin/python .claude/outputs/figure-layout-fix/v5f2_tall.py     # 4.950 x 2.250 in
FRP_OUT=<dir> .venv/bin/python .claude/outputs/figure-layout-fix/v5fC2_tall.py    # 4.730 x 3.550 in
FRP_OUT=<dir> .venv/bin/python .claude/outputs/figure-layout-fix/v2f3_tall.py     # 4.400 x 3.600 in
.claude/outputs/figure-layout-fix/build.sh <variant-dir>                          # pdflatex x3
.venv/bin/python .claude/outputs/figure-layout-fix/measure.py <pdf> "main text=References"
```
All five scripts take `FLF_W` / `FLF_H` environment overrides (printed width / height in inches), so any re-size the Advisor asks for is a one-line change.

---

## Open questions / follow-ups / risks

1. **The 9→10 pp decision is the Advisor's, and it is a genuine either/or.** §6.1 row iii is the load-bearing measurement: legible Fig C.2 alone lands END at 9.00 pp with 0.45 pt to spare. There is no configuration with both appendix figures legible at 9 pp. If 9 pp is truly hard, the only remaining lever is **dropping panels from Fig C.2**, which is a content call (rule 3) I did not take.
2. **The V5 BUILD-NOTE §6 width claim does not reproduce** (§3.3): `0.70\linewidth` at the banked aspect builds 9 pp / 4.00 pp today, where the note says 0.62–0.84 all spill. Someone should re-check or annotate that line before it is cited again.
3. **V5 Fig 1's "split" is not the split the task described** (§4.1) — the figure was never multi-panel. I executed the defensible analogue (reduce + full-size annotated companion in the appendix) and am flagging the mismatch rather than pretending the prescription applied. If the Advisor prefers *no* appendix companion, deleting the l.157–162 `figure` block returns V5 to build C (still 10 pp, END 9.35).
4. **The main figure has one un-decoded mark** — the crimson colourbar tick (caption item N2). It is the one place where restoring a label was impossible inside the 1.42 in ceiling without going below 8 pt.
5. **No `chlu/` bug found**; nothing to hand `experiment-engineer`. The process footgun the render pass named (`scratch/v5-revision-1/fig1_collapse.py` and `regen_figs.py` write straight into the frozen `papers/v5-short/figs/`) is **still unfixed** — all five of my scripts take an `FRP_OUT` override precisely to avoid it, and I recommend the same override be added to those two.
6. **`papers/v2-neurreps-descoped` now ships new figures with un-edited captions**, contrary to its own BUILD-NOTE §10 instruction, because my task ordered the install and forbade caption edits. **That build must not be sent anywhere until §8 is worked.**

---

## Proposed handover updates (for the Hub)

**§1.6 / experiments — nothing to add: this pass made zero measurements.** All five renders are replots of banked artifacts and all five reproduce the original generators' tap digests bit-for-bit (§5).

**§5 provenance — add these, they are new facts:**
- **The V5 main-text figure-height budget is 1.42 in** (box height at any width; 1.44 in spills to 5 pp), measured by placeholder sweep at 0.45 / 0.60 / 0.70 `\linewidth`. The main text's last body line sits 17.84 pt above the text-block bottom. This is now a known, reusable number for any future V5 main-text figure decision.
- **The V2 de-scoped build's printed figure boxes are identical to the reframe's** (≤0.011 pt), so any typography verdict measured on one transfers to the other.
- **`papers/palm-variant/v5/BUILD-NOTE.md` §6's width claim (0.62–0.84 all spill) does not reproduce on the current source** — 0.70 builds 9 pp / 4.00 pp. Flag as stale.
- Generator map extension: the five new figures are regenerable from `.claude/outputs/figure-layout-fix/{v5f1_main,v5f1_appx,v5f2_tall,v5fC2_tall,v2f3_tall}.py`, all `FLF_W`/`FLF_H`/`FRP_OUT`-parameterised. The render pass's V2-Fig-2 gap (no surviving generator) is **unchanged and still open**.

**§8 / open items — three owned actions:**
1. **CAPTION EDITS OWED — the consolidated list is §8 of this report: 5 live items + 2 new, and 4 of the render pass's 9 are now moot.** Two are hard defects in the shipped build (V5 Fig 3's `I-J/I-R3/I-R1` naming; V5 Fig 4's "panels are labelled by the pre-registered prediction" clause). **Needs a curator/writer task.** ⛔ Includes signing off the caption I had to write for the new V5 Fig 2. Also re-key every downstream mention of "V5 Fig 2 / Fig C.2" to **Fig 3 / Fig 4**.
2. **PAGE-BUDGET DECISION OWED — V5 is now 10 pp** (main text held at 4.00 pp; proven not caused by the main figure). The trade table and the 9-pp fallback are priced in §6.1/§6.3. **Advisor's call.**
3. **NEW HYGIENE ITEM — V5 Tables 2 and 3 still print the internal instrument IDs** `I-J / I-R1 / I-R2 / I-R3`, `Γ_jac/Γ_R3`, `n_jac/n_R1` in their headers (§7). Same class as the figure labels the render pass stripped; not on any existing list.

**Also worth recording:** `measure.py` + `imgbox.py` (printed image boxes from `mutool` traces, page splits, last-page slack) are now checked-in-under-`.claude` instruments; combined with the placeholder-sweep method they turn "will this figure fit?" into a five-second measurement rather than an argument.
