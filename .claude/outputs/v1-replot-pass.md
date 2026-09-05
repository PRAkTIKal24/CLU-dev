# v1-replot-pass — results-analyst report

**Task + acceptance criterion:** re-render `fig1_certificate.png`, `fig4_bibo.png`, `fig_frontier_clean.png` with the Head's exact string changes and the Hopfield band removed, at printed-size-legible type, **proving with `tap.py` that no plotted value moved** and that no `.tex` file was touched.
**Status: done.** All three installed. **fig1 and fig4 tap digests are bit-identical to their predecessors** (`88273eb2…`, `ae96d8b3…`); the frontier differs by **exactly one call — the removed `axhspan` (Hopfield band)** — and by nothing else. Both PDFs rebuild with **page-by-page text identical** to the control.

⚠ **THIS REPORT CONTAINS A DOWNSTREAM RECONCILIATION LIST — IT NEEDS AN OWNER.** §8 lists **3 caption edits owed in `pj_sub.tex`**, ⛔ none of which I made (task forbids it).

⭐ **RESOLVED MID-PASS — the register drift is closed.** My first render used the task's original mandated annotation *"squeeze priced out of the swept ζ ≤ 2.0"*, which I flagged because the Head's concurrent prose pass had meanwhile taken `priced out` from 3 → **0** occurrences in `pj_sub.tex`. **The Head amended §3a of the task file to `squeeze: beyond the swept ζ ≤ 2.0`, and the shipped PNG now carries that string.** Verified: `priced out` = **0** in the generator, **0** in `pj_sub.tex`. ⛔ At no point did I substitute my own wording.

**DIAL DECLARATION (echoed).** **Dials touched: NONE** — instrument/figure regeneration from banked artifacts. **Laundering control:** n/a (no performance number produced). **Falsifies:** n/a. **Does NOT falsify:** n/a. No experiment run, no config changed, no `.tex` edited, and — proven with the data tap — **no plotted value changed**.

**Pre-registration rule:** does not bind. The acceptance criterion is a **hash equality**, not a measured ratio/exponent/slope/law. The prediction I did commit to before running (*"re-running the surviving generator must reproduce the banked PNG byte-for-byte, and the restyled render must produce an identical tap multiset"*) is reported below as a pass/fail, not a fitted quantity. No `PREREG.md` written.

---

## 0. Environment / provenance

| | |
|---|---|
| repo HEAD | `7fcef501fb1aeae33a8d149b046c9b6126dcecd7` (tracked tree **not** modified by this pass) |
| interpreter | main venv `/Users/user/Desktop/CHLU/.venv` (Python 3.11.13). **No JAX needed** — all three figures are replots of banked JSON. |
| libs | **matplotlib 3.10.8, numpy 2.4.1, Pillow 12.1.0** (matches the `Software` tag inside every banked PNG) |
| TeX | pdfTeX 3.141592653-2.6-1.40.29, **TeX Live 2026** (`/Library/TeX/texbin/pdflatex`), `mutool` + `pdftotext` from `/opt/homebrew/bin` |
| scratch | `.claude/scratch/v1-replot-pass/` · artifacts `.claude/outputs/v1-replot-pass/` |
| git footprint | **none.** No tracked file created, modified or committed. No branch, no worktree. |

**Flag-provenance of the plotted numbers (inherited — this pass measured nothing new).**

| figure | source artifact | seeds / n | non-default config in effect at measurement time |
|---|---|---|---|
| fig1 (a) reach | `.claude/outputs/v1-certificate-payoff/paid_access_metrics.json` → `reach` | **5 seeds**, oracle placement | symmetric double-well, **dim 2**, causal box **L = 2.5**, swept rapidity budget **ζ ≤ 2.0**, distances `d ∈ {0.8,1.6,2.4,3.2,4.0,5.0}`, arms {wormhole, state-replacing map, Newtonian squeeze control, squeeze S⁽ᴹ⁾, dense/throat-V, plain relaxation} |
| fig1 (b) latch | same JSON → `certificate_payoff.latch` | 16 sample states from the capture ball | det J measured by forward-mode AD: wormhole/random-shift **1.0**, state-replacing map **0.0**; `std(Q_in) = 0.08028798550367355`; `std(Q_out)` wormhole-across-coset **0.08028798550367355**, state-replacing **0.0** (single value 1.5) |
| fig4 BIBO | same JSON → `certificate_payoff.bibo` (12 entries) | — | non-coercive `V = ½kq₀² − εq₀⁴`, coercive edge **x_b = 3.54**, `r*` over **2T** steps, exit distances `b ∈ {1.0,2.0,3.0,3.6,4.0,5.0}`; `r*(2T)` blind at b=5.0 = **104.833**, certified = **0.0908** |
| frontier | `.claude/scratch/regime-remap-2000ep/runs/*.json`, cells `correlation_N{128,256,384}_kv{32,64,96,128}_s0.0_seed4{2,3,4}_ep{500,1000,2000,4000}_ne3_v{256,512}` | **n = 3 seeds {42,43,44}, ne3** | corr ρ = 0, stress 0.0, vocab 256 (kv128 → 512); aggregation = mean ± population std over the 3 seeds |

⚠ **The frontier is n = 3, not the 8 pooled seeds quoted elsewhere in the paper.** The left-panel title retains `3 seeds` as instructed; the caption must state **n = 3** (§8).

---

## 1. Backups and the `.tex` no-touch proof

`MANIFEST_BEFORE.txt` / `MANIFEST_AFTER.txt` (both in `.claude/outputs/v1-replot-pass/`):

| file | md5 BEFORE | md5 AFTER | verdict |
|---|---|---|---|
| `figs/fig1_certificate.png` | `679647f639bfb8b3b7ecfa1333f43b69` | `47adc87bf09ccef4fbf08df504584677` | **replaced** |
| `figs/fig4_bibo.png` | `708b6fae2dadc755291b07c2962d102f` | `5fdd6bd2f2dcf06a4615cc491023ac7d` | **replaced** |
| `figs/fig_frontier_clean.png` | `bcc5f32dcd85e01740638c6608f26320` | `2a217098434d80a15e98a12419109182` | **replaced** |
| `figs/fig2_regime_map.png` | `b0cfbf53651ac187bacee0f977d93f1e` | `b0cfbf53651ac187bacee0f977d93f1e` | ✅ untouched |
| `figs/fig_regime_map.png` | `8b1dfddc54b4e48da0254a3bf35b9159` | `8b1dfddc54b4e48da0254a3bf35b9159` | ✅ untouched |
| `figs/paid_access_reach.png` | `fc372ae54cf39d7181f68837ad0e463b` | `fc372ae54cf39d7181f68837ad0e463b` | ✅ untouched |
| `submission.tex` | `caef2272f9dc96d349b46486563d24ee` | `caef2272f9dc96d349b46486563d24ee` | ✅ **untouched** |

⛔⛔ **`pj_sub.tex` md5 DID change during my session — `de3585a6794add42c657600c9aa022db` → `08d31733b5648ed6ab4a6bbc5dc07ed8` — and it was NOT me.** Proof it was the Head's live prose pass, not this pass:
1. I never wrote to that path. All builds ran on **copies** under `.claude/scratch/v1-replot-pass/build/{base,ctl,new}/`. Immediately before and immediately after each of my two `cp … figs/` installs I printed the md5: **identical across the install** (`08d31733…` → `08d31733…` both times).
2. `diff` of my spawn-time snapshot against the current file is **118 lines of prose**, all of it the register retirement the task predicted: `ledger` **19 → 0**, `receipt` **9 → 0**, replaced by `energy change` ×14 and `certificate` ×n; title changed *Paid Access* → *Certified Access*. Not one `\includegraphics`, figure environment or `figs/` path is in the diff.
3. Snapshots kept: `.../backup_figs/pj_sub.tex` (spawn, `de3585a6`) and `.../backup_figs/pj_sub_HEADEDIT_08d31733.tex`.

---

## 2. Provenance: every one of the three figures now has a generator that reproduces the banked PNG **byte-for-byte**

| figure | generator | re-run reproduces the shipped PNG? |
|---|---|---|
| `fig1_certificate.png` | `.claude/scratch/v1-revision-2/make_figs.py` (live, `OUT` repointed to scratch) | **YES — md5 `679647f6…` exact** |
| `fig4_bibo.png` | same file | **YES — md5 `708b6fae…` exact** |
| `fig2_regime_map.png` (not touched, run only as a control) | same file | **YES — md5 `b0cfbf53…` exact** |
| `fig_frontier_clean.png` | ⭐ **no generator existed — RECOVERED by reconstruction:** `.claude/outputs/v1-replot-pass/orig_frontier.py` | **YES — md5 `bcc5f32dcd85e01740638c6608f26320` exact, 0 differing pixels** |

⭐ **The lost frontier generator is now recovered, not merely approximated.** An unbounded per-file sweep for `frontier_clean` over `.claude/**` returned **17 files, all `.md`/`.tex`, zero `.py`** — confirming no script anywhere writes that filename (`analyze.py` writes the different, earlier `fig_frontier.png`; the two are **not** the same figure). I re-derived it from `runs/*.json` and closed the last three unknowns by pixel-diffing against the banked PNG:

| unknown | first guess | diff vs banked | resolved value | diff |
|---|---|---|---|---|
| right-panel marker | `"o"` | 2 332 px, max ΔRGB 517 | **`"s"`** (left panel really is `"o"`, right really is `"s"`) | 1 261 px, max 18 |
| Hopfield band style | `color="0.6", alpha=0.3` | 1 261 px on the band edge + legend swatch only | **`color="k", alpha=0.12`** | **0 px** |
| band bounds | report's rounded `0.947–0.976` | 2 650 px | **exact `(0.9461805555555555, 0.9756944444444443)`** = min/max of `hop_acc` over the 16 ne3 frontier cells | **0 px** |

Aggregation identified, not guessed: **only the `episodes == 3` subset** reproduces the figure (`kv32@500 fid = 0.7396 ± 0.1031`, matching the plotted point and the `regime-remap-2000ep.md` §Item-2 table). `tables.md` §Item 2 quotes **`0.76 ± 0.09`** for the same cell because it pools ne3 **and** ne5 seeds — ⚠ **the task's pointer to `tables.md` would have given the wrong numbers; I used the runs instead and validated byte-identity.**

---

## 3. ⭐ The validation bar: `tap.py` digests, old vs new

`tap.py` hashes every numeric positional argument of `Axes.{plot,loglog,semilogx,semilogy,scatter,bar,barh,errorbar,fill_between,axhline,axvline,axhspan,axvspan,step,stairs,hist,imshow,pcolormesh,contour,contourf}` as `<f8` SHA-1; text, labels, colours, sizes, limits and ticks are deliberately **not** recorded; comparison is an order-insensitive multiset (`compare.py`). `savefig` was redirected so no banked artifact could be overwritten.

| figure | ORIGINAL digest | NEW digest | data calls | verdict |
|---|---|---|---|---|
| **fig1_certificate** | `88273eb233a356633d565825aacd7351fd9f9343` | `88273eb233a356633d565825aacd7351fd9f9343` | 13 vs 13 | ✅ **IDENTICAL DATA** (whole-run digests equal, not merely the multiset) |
| **fig4_bibo** | `ae96d8b34e077002514d2a36e7a14e08c46329f9` | `ae96d8b34e077002514d2a36e7a14e08c46329f9` | 5 vs 5 | ✅ **IDENTICAL DATA** (whole-run digests equal) |
| **fig_frontier_clean** | `0f98906c1f893e2af6628a45ff3b200732d29a02` | `e02915edfab102311999edffb248c2908ad55fe2` | 9 vs 8 | ✅ **only difference = `axhspan:6a6a1cb454273d33,bbe0be3c7ea471cd × 1`** — the Hopfield band, whose removal is the task. `in NEW not in ORIGINAL : 0`. All **8** CLU curve calls identical. |

Digests: `.claude/outputs/v1-replot-pass/digests/{orig,new}_{fig1,fig4,frontier}.json` (+ `orig_fig2.json`).

⚠ **Tap blind spot, closed by hand.** `errorbar(x, y, yerr=…)` passes `yerr` as a **keyword**, so the tap does not hash it. The frontier's error bars are therefore covered by a separate proof: both `orig_frontier.py` and `new_frontier.py` call the **same** `frontier_data.load()`, and `orig_frontier.py` reproduces the banked PNG with **0 differing pixels**, so the error bars are provably the banked ones. For the record the full array set hashes to **sha1 `e78ab0834fde1941`**:

| kv | fidelity (500/1000/2000/4000) | ± std | gated acc | ± std |
|---|---|---|---|---|
| 32 | 0.739583 0.996528 1.000000 0.989583 | 0.103120 0.004910 0.000000 0.008505 | 0.298611 0.982639 0.996528 0.892361 | 0.035410 0.017705 0.004910 0.048362 |
| 64 | 0.411458 0.819444 1.000000 0.998264 | 0.017010 0.038587 0.000000 0.002455 | 0.057292 0.420139 0.991319 0.965278 | 0.019488 0.021404 0.004910 0.006496 |
| 96 | 0.387731 0.240741 0.969907 0.998843 | 0.022915 0.064046 0.012784 0.001637 | 0.023148 0.107639 0.916667 0.974537 | 0.003274 0.014175 0.022143 0.004331 |
| 128 | 0.397569 0.092014 0.705729 0.999132 | 0.021298 0.011711 0.059536 0.001228 | 0.010417 0.035590 0.597222 0.949653 | 0.003683 0.004426 0.059092 0.012276 |

Removed Hopfield band = **[0.9461805555555555, 0.9756944444444443]**.

---

## 4. Before → after string table (every text element that changed)

### 4a. `fig1_certificate.png`
| where | BEFORE (shipped) | AFTER (rendered) | mandated? |
|---|---|---|---|
| ⛔ panel (a) annotation | `squeeze collapses\npast the box` | `squeeze: beyond\nthe swept ζ ≤ 2.0` | **yes, verbatim** — this is the Head's **amended** §3a string (task file revised mid-pass; the superseded target was `squeeze priced out of the swept ζ ≤ 2.0`, rendered once and replaced). Line break inserted to fit; wrap is permitted by §3. |
| panel (a) title | `(a) Reach: who lands — and with which receipt` | `(a) Landing rate vs. basin distance` | yes |
| panel (b) title | `(b) The receipt cashed out: latch transported vs erased` | `(b) Goldstone charge: transported vs. erased` | yes |
| (a) legend | `wormhole  (det J = 1, ledger = 0)` | `wormhole  (det J = 1, ΔV = 0)` | yes |
| (a) legend | `no-physics router  (det J = 0)` | `state-replacing map  (det J = 0)` | yes |
| (b) legend | `no-physics router (det J=0)` | `state-replacing map (det J = 0)` | yes |
| (b) inset box | `router:   std(Q_out) = 0.0  (erasure)` | `state-replacing: std(Q_out) = 0.0  (erasure)` | yes |
| ✅ unchanged | `causal box L = 2.5` · `crossover bracket` · `basin distance d` · `landing rate  (5 seeds)` · `incoming/outgoing charge` · `identity (Q preserved)` · `random shift (det J=1, no channel)` · `std(Q_in) = 0.0803` · `wormhole: std(Q_out) = 0.0803  (transport)` · `wormhole, coset-tangent (det J=1)` · `wormhole, across-coset (det J=1)` · `Newtonian squeeze (control, det J = 1)` · `squeeze S⁽ᴹ⁾  (det J = 1)` · `dense / throat-V  (no jump)` · `plain relaxation` | — | — |

### 4b. `fig4_bibo.png`
| where | BEFORE | AFTER | mandated? |
|---|---|---|---|
| title | `BIBO: an uncertified exit escapes; the receipt refuses it` | `Maximum excursion radius vs. destination\nlocus of the wormhole jump` | **yes, verbatim** (2-line wrap — one line is 332 pt against a 273 pt axes) |
| legend | `wormhole, receipt ignored (ablation)` | `wormhole, screen ignored (ablation)` | yes |
| legend | `no-physics router (coincides with ablation)` | `state-replacing map (coincides with ablation)` | yes |
| legend | `wormhole + receipt (screened; refuses)` | `wormhole + screen (refuses exit)` | yes |
| red annotation | `b = 5.0: energy ledger ΔH = 0 (FREE)\n— and the blind exit still escapes` | `b = 5.0: energy change ΔH = 0 (free)\n— the unscreened exit still escapes` | yes |
| ✅ unchanged | `coercive edge x_b = 3.54` · `escape radius` · `requested exit distance  b` · `r* = max_t ‖q_t‖  over 2T steps` | — | — |

### 4c. `fig_frontier_clean.png`
| where | BEFORE | AFTER |
|---|---|---|
| right panel | shaded Hopfield band + `Hopfield band` legend entry | ⛔ **both removed** (the single `axhspan` in §3) |
| right title | `Gated acc vs epochs (Hopfield band shaded)` | `Gated acc vs epochs` |
| left title | `Fidelity vs epochs (corr=0, ne3, 3 seeds)` | `Fidelity vs epochs\n(corr=0, ne3, 3 seeds)` — ✅ **`3 seeds` kept**, 2-line wrap only |
| ylabels | `CLU-EBM storage fidelity` / `CLU gated accuracy` | same strings, wrapped to 2 lines |
| ✅ kept | both panels, all four CLU curves (kv32/64/96/128) in **both** fidelity and gated accuracy, all error bars | — |

**String sweep, positive-controlled.** In the ORIGINAL generators: `collapses`×1, `no-physics router`×3, `ledger`×3, `receipt`×7, `Hopfield band`×2. In the NEW generators: **`collapses` 0 · `no-physics router` 0 · `ledger` 0 · `receipt` 0 · `FREE` 0 · `blind exit` 0**; `Hopfield band`×1 and that single hit is in `new_frontier.py` **line 2, the module docstring** ("…Hopfield band REMOVED…") — never handed to matplotlib. Required new strings all present ≥1 (incl. the amended `squeeze: beyond` ×1); **`priced out` ×0 in the generator and ×0 in `pj_sub.tex`.** ⇒ **`collapses past the box` appears in no PNG; `no-physics router` appears in no PNG.**

---

## 5. Printed-box measurements (from **built PDFs**, `mutool draw -F trace`, 72 pt/in)

Control build = the **current** `pj_sub.tex` + `submission.tex` with the **old** PNGs restored; new build = same `.tex`, new PNGs. Both built ×3 with `pdflatex -interaction=nonstopmode`.

| figure | include | printed box BEFORE | printed box AFTER | **Δ** |
|---|---|---|---|---|
| fig1 (`pj_sub.pdf` p4 / `submission.pdf` p5) | `width=\textwidth` | 468.006 × 169.841 pt | 468.003 × 169.840 pt | **−0.003 × −0.001 pt** |
| fig_frontier_clean (`submission.pdf` p11) | `0.72\linewidth` | 336.960 × 129.600 pt | 336.967 × 129.603 pt | **+0.007 × +0.003 pt** |
| fig4_bibo (`submission.pdf` p19) | `0.68\linewidth` | 318.246 × 212.164 pt | 318.235 × 212.157 pt | **−0.011 × −0.007 pt** |
| fig2_regime_map (untouched, control) | `width=\textwidth` | 467.996 × 134.159 pt | 467.996 × 134.159 pt | 0 |

All deltas are **< 0.015 pt** — pure integer-pixel rounding of the canvas (each new PNG is an exact integer multiple of the banked aspect: 2604×945 = 2480:900, 1767×1178 = 1320:880, 1872×720 = 1560:600). Pagination consequences, measured not assumed:

- `pj_sub.pdf`: **14 pages both**; `pdftotext` page-by-page output **byte-identical**; `pagesplit.py` → `References ends at 12.59 pp`, `END 14.00 pp` **in both**; overfull 1 / underfull 5 / undefined 0 **in both**.
- `submission.pdf`: **25 pages both**; `pdftotext` page-by-page output **byte-identical**; overfull 3 / underfull 17 / undefined 0 **in both**.

## 6. Type at printed size

Every figure is now rendered on a canvas whose width **is** its printed width, so the specified pt **is** the printed pt (scale ≈ 1.000). "Before" = the generator's pt × the measured shrink.

| figure | shrink BEFORE | ticks (≥7) | axis labels (≥8) | titles (≥9) | legend (≥8) | in-figure annotations |
|---|---|---|---|---|---|---|
| **fig1** | 6.5001/12.4 = **0.5242×** | 5.24 → **7.01** ✅ | 5.24 → **8.01** ✅ | 5.77 → **9.01** ✅ | 3.88 / 3.98 → **8.01** ✅ | 4.46 / 4.19 / 3.88 → **7.01** ✅ (incl. the monospace stats box) |
| **fig4** | 4.4201/6.6 = **0.6697×** | 6.70 → **7.00** ✅ | 6.70 → **8.00** ✅ | 7.37 → **9.01** ✅ | 5.09 → **8.00** ✅ | 5.36 → **7.00** ✅ |
| **frontier** | 4.68/13 = **0.3600×** | 3.60 → **7.00** ✅ | 3.60 → **8.00** ✅ | 4.32 → **9.00** ✅ | 3.60 → **8.00** ✅ | — |

⚠ fig1 is specified at 9.02/8.02/7.02 pt deliberately: its canvas is 6.5100 in against a 6.5000 in printed box (the 1-px aspect rounding), a 0.99847× scale that would otherwise land ticks at 6.99 pt. **All three figures now clear every target strictly.**

Fit diagnostics at printed size (`fitcheck.py`): fig1 axes 189.4 × 125.2 pt with legends at **91.6 %** (a) and **81.6 %** (b) of axes width; fig4 axes 273.5 × 158.7 pt, legend **73.9 %**; frontier axes 118.8 × 67.1 pt, legend **34.2 %**. No artist is clipped by the canvas on any figure (`unclip()`; verified independently — **0 ink pixels on all four edges of all three PNGs**).

**Layout changes made to buy the type (no data touched):** fig1 (a) legend 2-col → 1-col with `ylim` top 1.68 → 2.50 to seat it; `causal box L = 2.5` wrapped to two rotated lines; fig1 (b) `ylim` opened to (0.55, 2.30) so the enlarged legend and stats box clear every point; fig4 title wrapped to 2 lines; frontier legends 1-col lower-right, ylabels/left title wrapped, and the crowded `500` x-tick label dropped to a second row (at 7 pt, `500` and `1000` are 15.6 pt of glyph in a 15.8 pt gap).

Before/after sheets rasterised **at the printed width** (300 dpi) — this is what a referee sees:
`.claude/outputs/v1-replot-pass/compare_{fig1_certificate,fig4_bibo,fig_frontier_clean}.png`

## 7. `fig_regime_map.png` and `paid_access_reach.png` — left alone

✅ **Confirmed untouched.** Both md5s are identical before and after (§1). Neither was re-rendered, opened for writing, or copied over. `fig2_regime_map.png` is likewise byte-identical: it *was* regenerated by the original generator into a **scratch** directory as a provenance control (§2) and that scratch copy matched the shipped one exactly, but nothing was written into `figs/`.

---

## 8. ⛔ Caption / annotation edits owed — **FOR THE HEAD, NOT MADE BY ME**

0. ✅ **CLOSED, no action.** The `priced out` drift (task §3a's original annotation vs. `pj_sub.tex` where the Head's live pass took `priced out` 3 → 0) was resolved by the Head amending §3a to **`squeeze: beyond the swept ζ ≤ 2.0`**. Re-rendered and re-installed; `priced out` now appears in **neither** the PNG nor the `.tex`. ⚠ One residual check for the Head: the caption at `pj_sub.tex:64` reads *"stepping up until the required energy exceeds the swept rapidity budget ζ ≤ 2.0"* — the annotation's *"beyond the swept ζ ≤ 2.0"* is consistent with it, but the caption is the place where "beyond **what**" (energy, not capability) is spelled out, so the two should be read together once more.
1. **Figure 1 caption, `pj_sub.tex:64`, panel (b) label.** The caption says **"(b) The certificate cashed out:"**; the rendered panel-(b) title is now **"(b) Goldstone charge: transported vs. erased"**. Suggest aligning the caption's lead-in to the panel title.
2. **Figure 1 caption, panel (a) shading.** The caption says the squeeze's theoretical reach `[L, L+p₀ sinh ζ/M₀]` is "**(shaded)**". The only shading in panel (a) is the grey span **L → 3.4** labelled *crossover bracket* (unchanged, and correct per the data). If "(shaded)" is meant to point at that span, it should say so; otherwise the parenthetical points at an artefact the figure does not carry.
3. **Frontier figure caption must state n = 3.** MUST-FIX inherited from the fidelity audit: the frontier is **3 seeds {42,43,44}, ne3** — not the 8 pooled seeds quoted elsewhere. Suggested clause: *"…epoch-scaling frontier at corr = 0, mean ± s.d. over **n = 3 seeds** (42–44, 3 episodes/cell); this is a different seed set from the n = 8 pooled figures in §4.1."* The panel title already carries `3 seeds`.

**Insertion-width rule for the two figures not currently in `pj_sub.tex`** (§7 open items 1–2). Each canvas *is* its printed box, so effective type scales linearly with the include width:
- **fig4_bibo** — canvas 4.4175 in. Insert at **≥ 0.68\linewidth (4.42 in)** to keep ticks ≥ 7 pt. At `\textwidth` the type becomes 1.47× (10.3 pt ticks — legible but over-large); below 0.68\linewidth it falls under target.
- **fig_frontier_clean** — canvas 4.6800 in. Insert at **≥ 0.72\linewidth (4.68 in)**.
- **fig1_certificate** — canvas 6.5100 in, already at `width=\textwidth`; **do not shrink it**.

## 9. Two open items — reported, not acted on

1. **`fig2_regime_map.png` is not in `pj_sub.tex`** and I did not re-render it. Its panel (a) (storage fidelity, 500 vs 2000 ep) is CLU-internal and could return as a single-panel figure; its panels (b) and (c) are the Hopfield scoreboard and would re-open the comparison the paper removed. ⛔ **Awaiting the Head's word.** If asked, it is a 5-minute job: its generator is live and reproduces byte-identically, and a single-panel cut of (a) would be a strict subset of a verified digest.
2. **`fig4_bibo.png` is not in `pj_sub.tex`.** It is now rendered and ready; the Head inserts it (width rule above).

## 10. Limitations / risks

- **The frontier's provenance rests on a reconstruction, but a *verified* one** — 0 differing pixels against the banked PNG, which is the same class of evidence as a surviving generator. What is *not* verified is that the banked PNG's own aggregation was intended; I only proved it is `episodes == 3`, n = 3. ⚠ `tables.md` §Item 2 (the source the task pointed me at) quotes `0.76 ± 0.09` for kv32@500 against the figure's `0.74 ± 0.10` because it pools ne3 + ne5 — **anyone re-deriving this figure from `tables.md` will get different numbers.**
- **`errorbar(yerr=…)` is invisible to `tap.py`** (kwarg, not positional). Closed for the frontier by byte-identity + an explicit array hash (§3); fig1 and fig4 have no error bars.
- **fig4 and the frontier were footprint-matched to `submission.tex`**, the only file that currently includes them. If the Head inserts them into `pj_sub.tex` at a different width, the type table in §6 rescales by that ratio (§8 rule).
- **`pj_sub.tex` moved under me mid-pass** (§1). My printed-box and pagination control/new builds both used the *current* (`08d31733`) file, so the comparison is internally consistent; but if the Head edits further, the box numbers should be re-measured — the instrument is `/tmp/boxes.py` (copied to nothing tracked; the two-line `mutool draw -F trace` parse is reproduced in this report's §5 heading).
- **No dial was tested and no performance number was produced.** Nothing here is evidence for or against any CHLU claim; it removes a figure–text contradiction and makes three figures legible.

---

## Proposed handover updates (for the Hub)

**§1.6 / experiments — nothing to add.** No experiment was run; no measured claim changed.

**§5 provenance — add:**
- ⭐ **The lost generator for `fig_frontier_clean.png` is RECOVERED**, not approximated: `.claude/outputs/v1-replot-pass/orig_frontier.py` + `frontier_data.py` reproduce the banked PNG (`md5 bcc5f32dcd85e01740638c6608f26320`) with **0 differing pixels**. Recovered parameters worth banking: figsize (13, 5) @ dpi 120; left panel marker `"o"`, **right panel marker `"s"`**; Hopfield band `axhspan(color="k", alpha=0.12)`; band bounds are the **exact** min/max of `hop_acc` over the 16 ne3 frontier cells = **[0.9461805555555555, 0.9756944444444443]**, *not* the rounded 0.947–0.976 in the prose.
- ⚠ **`regime-remap-2000ep/tables.md` §Item 2 and the figure disagree by construction.** `tables.md` pools ne3 **and** ne5 (kv32@500 → `0.76 ± 0.09`); the figure and `regime-remap-2000ep.md` §Item 2 use **ne3 only, n = 3, seeds {42,43,44}** (kv32@500 → `0.7396 ± 0.1031`). Any future task that says "reconstruct from `tables.md`" will produce wrong numbers. Full 32-array set hashes to **sha1 `e78ab0834fde1941`** (values tabulated in this report §3).
- All three V1 figures now have generators banked in `.claude/outputs/v1-replot-pass/` (`new_fig1.py`, `new_fig4.py`, `new_frontier.py`) with the tap digests that certify them (`digests/`).

**§8 open directions / paper state — add:**
- ✅ **The MF-B claims regression in the headline figure is CLOSED.** `squeeze collapses past the box` and `no-physics router` no longer appear in any shipped PNG (positive-controlled sweep, §4).
- ⭐ **A second, subtler register drift opened AND closed inside the same wave, and the mechanism is worth banking.** The Head's live prose pass took `priced out` from 3 → 0 in `pj_sub.tex` *while* the task file still mandated *"squeeze priced out of the swept ζ ≤ 2.0"* as the replacement annotation. The spoke rendered the mandated string verbatim, flagged the contradiction rather than substituting, and the **Head amended the task file mid-pass** to `squeeze: beyond the swept ζ ≤ 2.0`; the figure was re-rendered with digests unchanged. ⚠ **Lesson for concurrent prose/figure passes: a figure task's mandated strings are a snapshot of the prose and go stale if the prose pass runs in parallel — the string table needs a re-check at install time, not only at scoping.**
- **3 caption edits are owed in `pj_sub.tex`** (report §8) and **2 figures are still un-inserted** (`fig4_bibo`, `fig_frontier_clean`) with a stated minimum include width.
- **Frontier figure is n = 3, not n = 8.** This must reach the caption; it is a MUST-FIX inherited from the fidelity audit and is still unwritten in `pj_sub.tex`.

**Code bug for `experiment-engineer`:** none found in `chlu/`. Two **instrument** notes: (i) `.claude/scratch/figure-render-pass/tap.py` does not hash `errorbar`'s `yerr`/`xerr` (keyword-only) — worth a 3-line fix before the next replot pass leans on it for a figure with error bars; (ii) `.claude/scratch/v1-revision-2/make_figs.py` still hard-codes `OUT = …/.claude/papers/v1-short`, so an unmodified run writes into the canonical paper directory — worth making it an env var.
