# v2-figure-text-pass — results-analyst report

Task + acceptance criterion: execute the Head's two figure TODOs on `.claude/NIPSsubmission/v2-neurreps/figs/` — retire "decade" vocabulary from every figure that renders it, and relabel `fig_lifetime_headtohead.png` ("published" → **"single-exponential estimator"**, "5 trained models" → "CLU (5 seeds)") — re-rendering from generators, with plotted data provably unchanged, footprints preserved, and zero writes outside `figs/` and scratch.

**Status: done.** Both changes executed; **2 of 5 PNGs re-rendered, 3 swept clean and left byte-identical**; **every plotted array hash identical before/after** (tap digests below); **every printed image box identical to the 0.001 pt**; **all 8142 word bounding boxes in the built PDF identical between control and new builds** (zero pagination movement); zero writes to `pj_sub.tex`/`pj_sub.pdf`/`.aux`/`.log`.

**⚠ THIS REPORT CONTAINS A 3-ITEM RECONCILIATION LIST THAT NEEDS AN OWNER (protocol §5 corollary) — see §8.** The headline item: **I changed a THIRD "published" site the task did not enumerate** (an in-axes annotation), §3.2 item (d). One-line revert is given. Second: **two of the five figures are below the type targets at printed size — inherited, not caused here** (§6). Third: the `\TODO` at l.95 instructs a *citation number*, which the Head has since overruled in favour of a descriptive label — whoever deletes the tag must not act on its literal text (§7).

**DIAL DECLARATION (echoed): none — a text-on-figure pass. Laundering control: n/a. Falsifies the claim: n/a. Does NOT falsify: n/a. ZERO new measurements; ZERO data changes.**
**Pre-registration rule: not triggered** — no measured ratio, exponent, slope or law is produced by this task. (The identity checks below are byte/hash equalities, not estimates.)

**String confirmation requested by the mid-flight correction: I rendered `single-exponential estimator` (with the "-ial"). The superseded `single-exponent` form appears nowhere in any shipped PNG or generator.**

---

## 1. Provenance: which generator makes each shipped PNG

The five PNGs in `.claude/NIPSsubmission/v2-neurreps/figs/` are **byte-identical to `.claude/papers/plain/v2/figs/`** (md5 match on all five), i.e. this submission folder inherits the *plain build*'s figure set, not the `neurreps-variants/v2` set. Generators traced accordingly:

| shipped PNG | generator (executed) | provenance evidence |
|---|---|---|
| `fig1_gmor.png` | `.claude/scratch/plain-text-builds/plain_v2_gmor.py` → `fig1_gmor()` only | **re-run reproduces the shipped PNG byte-for-byte**, md5 `6e03247a…`, sha1 `5738a30e…` |
| `fig_lifetime_headtohead.png` | `.claude/scratch/plain-text-builds/plain_v2f1.py` | **re-run reproduces byte-for-byte**, md5 `8ea8afcf…`, sha1 `6eb1422d…` |
| `fig2_anchor_cure_laws.png` | `.claude/scratch/figure-render-pass/new_v2f2.py` (reconstruction; no original survives) | not re-run — **zero text hits, file untouched** |
| `fig3_gmor_condensate.png` | `.claude/scratch/figure-render-pass/new_v2f3.py` | not re-run — **zero text hits, file untouched** |
| `fig3_retention_overlay.png` | `.claude/scratch/v2-prefreeze-baselines/make_figures.py` → Fig A block (= banked `outputs/v2-prefreeze-baselines/figA_retention_overlay.png`, md5 `2cd50bd2…`, exact match) | not re-run — **zero text hits, file untouched**. ⚠ Generator needs JAX **and trained checkpoints** (`clu_drift_curve` → `load_run`); had it needed a text change it would have been the expensive one. |

Backups: `.claude/scratch/v2-figure-text-pass/figs-BEFORE/` (all five, md5+sha1 manifest `figs-BEFORE.md5`); post-state in `figs-AFTER/` + `figs-AFTER.md5`. Any figure is one `cp` from restoration.

## 2. Change 1 — "decade" sweep, all five figures (auditable, includes the zero-hit figures)

Method: (i) grep every generator for `decade` (case-insensitive); (ii) **visual read of all five rendered PNGs** (titles, axis labels, tick labels, legends, in-axes annotations) — the authoritative check, since a raster PNG cannot be grepped for rendered text.

| figure | rendered "decade" hits | where | disposition |
|---|---|---|---|
| `fig1_gmor.png` | **1** | right-panel title, line 2: `(max deviation ∼10⁻¹², all seeds, 4.5 decades)` | **changed** → `4.5 orders of magnitude` |
| `fig2_anchor_cure_laws.png` | **0** | — | **swept clean, untouched** (md5 unchanged `87dd174a…`) |
| `fig3_gmor_condensate.png` | **0** | — | **swept clean, untouched** (md5 unchanged `d2e045ad…`) |
| `fig3_retention_overlay.png` | **0** | — | **swept clean, untouched** (md5 unchanged `2cd50bd2…`) |
| `fig_lifetime_headtohead.png` | **0** | — | swept clean for "decade" (changed for reason 2 only) |

Generator-level positive control: `grep -ic decade` fires **1** on `plain_v2_gmor.py`, **0** on the other four generators, **3** on the task file.
`decade` in `pj_sub.tex`: only inside the `\TODO` at l.81 (confirmed) — no caption carries it.

## 3. The before → after string table (every string changed, and nothing else)

### 3.1 `fig1_gmor.png` — 1 string literal
| | text |
|---|---|
| before | `GMOR spectral-mass law: exact on the learned vacuum\n(max deviation $\sim 10^{-12}$, all seeds, 4.5 decades)` |
| after | `GMOR spectral-mass law: exact on the learned vacuum\n(max deviation $\sim 10^{-12}$, all seeds,\n4.5 orders of magnitude)` |

**Wrapped, not abbreviated, not shrunk** (method §3). Reason, measured: the phrase is longer than "decades", and this generator saves with `bbox_inches="tight"`, so on **one** line the tight bbox widened **1424 → 1450 px**; at `width=\linewidth` that shrinks the printed image from **147.111 → 144.473 pt** tall and shifts 130 words on one page by up to **1.90 pt**. Wrapping to a third title line keeps the canvas at **exactly 1424×529 px** (the third line fits inside the existing tight bbox because `tight_layout` re-allocates the right panel's axes height; the left panel's 2-line title still sets the bbox height). Measured alternatives, for the record:

| variant | canvas px | printed height @396 pt wide | word shifts vs control |
|---|---|---|---|
| **shipped** ("4.5 decades") | 1424×529 | 147.111 pt | — |
| **INSTALLED**: 3-line wrap, "4.5 orders of magnitude" | **1424×529** | **147.111 pt** | **0 (all 8142 words identical)** |
| alt A: 1 line, "4.5 orders of magnitude" | 1450×529 | 144.473 pt | 130 words, max \|Δy\| 1.90 pt, pagination unchanged |
| alt B: 1 line, "4.5 orders" (the permitted abbreviation) | 1424×529 | 147.111 pt | 0 |

The abbreviation "orders" was **not** needed and was not used.

### 3.2 `fig_lifetime_headtohead.png` — 4 string literals
| # | site | before | after |
|---|---|---|---|
| (a) | plot headline (title) | `Published lifetime protocol, run on trained CLUs` | `Single-exponential estimator lifetime protocol,\nrun on trained CLUs` |
| (b) | legend label (green band) | `published median 1.013` | `single-exponential\nestimator median 1.013` |
| (c) | legend label (curve proxy) | `5 trained models` | `CLU (5 seeds)` |
| (d) | **in-axes annotation, line 2** | `same pattern as the published $\epsilon=10^{-4}$ row` | `same pattern as the estimator's own $\epsilon=10^{-4}$ row` |

⚠ **(d) is a site the task did not enumerate.** The task named "the plot headline and the label"; the canvas carried the word **three** times. I changed it because the stated rationale for the whole relabel is that *"published" is factually wrong — the work is an unrefereed preprint* — and leaving the third instance would ship that exact false claim on the same canvas. It introduces **no author token and no citation number**. **One-line revert if the Head disagrees:** in `.claude/scratch/v2-figure-text-pass/gen/final_hth.py` l.55 restore `"same pattern as the published $\\epsilon=10^{-4}$ row"` and re-run (§4 command).

**Both relabels needed wrapping, and the wrapping was forced, not stylistic** (measured in §5):
- (a) On one line the title is **clipped at both ends** by the fixed canvas — the leading `S` and the trailing `CLUs` are cut off (evidence render: `renders/hth_new.png`). Wrapped to 2 lines.
- (b) At full width the legend box grows from 890 px to ~1450 px wide and **covers the EP peak** — the paper's headline feature. Quantified by a legend-occlusion instrument (§5): the unwrapped label hides **25.1 %** of the figure's coloured data ink. Wrapping after `single-exponential` brings occlusion to **exactly the shipped figure's value (zero)**.

## 4. Data-identity evidence (the acceptance criterion)

Instrument: the render pass's own data tap (`.claude/outputs/figure-render-pass/tap.py`) — SHA-1 of every numeric array passed to 20 wrapped `Axes` plotting methods, cast to `<f8`, compared as an order-insensitive multiset. Colours, labels, sizes, limits and ticks are deliberately **not** recorded. `savefig` is redirected so no banked artefact can be overwritten.

Command (both figures, both variants, one process):
`PYTHONPATH=/Users/user/Desktop/CHLU /Users/user/Desktop/CHLU/.venv/bin/python .claude/scratch/v2-figure-text-pass/gen/run.py gmor hth`
(`__file__` is pinned to the ORIGINAL generator paths so every relative path inside the scripts resolves exactly as it did when the shipped PNGs were made.)

| figure | variant | data calls | **tap digest** | PNG md5 | canvas px |
|---|---|---|---|---|---|
| `fig1_gmor` | orig (unmodified generator) | 7 | `9c284b32c84f341be86de43696c74e94212b6782` | `6e03247a0e5bee5ccad25ccf326315de` | 1424×529 |
| `fig1_gmor` | **final (installed)** | 7 | **`9c284b32c84f341be86de43696c74e94212b6782`** | `eaf697e6f18bb11342ff05b0189ccaa0` | **1424×529** |
| `fig_lifetime_headtohead` | orig | 9 | `baedf7981e42e7035a193dc996d375d53c9f9075` | `8ea8afcffc82ff6f8389cd23680d757c` | 2044×1118 |
| `fig_lifetime_headtohead` | **final (installed)** | 9 | **`baedf7981e42e7035a193dc996d375d53c9f9075`** | `932438846792c4016cd8af8280429ec8` | **2044×1118** |

- **Digest equality is exact, not multiset-approximate**: `tap digest equal = True` *and* `sorted(calls) equal = True` for both figures (no empty-proxy bookkeeping difference this time).
- **Tap fidelity proof**: running each *unmodified* generator under the tap wrote a PNG **byte-identical to the shipped file** (md5 `6e03247a…` and `8ea8afcf…` reproduced exactly) — so the tap is provably faithful to the artefact that ships today.
- Untouched figures need no tap: `fig2_anchor_cure_laws.png`, `fig3_gmor_condensate.png`, `fig3_retention_overlay.png` are **byte-identical before and after** (`figs-BEFORE.md5` vs `figs-AFTER.md5`), which is a strictly stronger statement than a digest match.
- Digests: `.claude/scratch/v2-figure-text-pass/renders/TAP_{gmor,hth}_{orig,final}.json`.

## 5. Legend-occlusion instrument (why the legend label is wrapped)

Coloured data ink = pixels with `max(RGB) − min(RGB) > 0.235` (the five seed curves, the green band, the purple EP line). Occlusion = ink(legend removed) − ink(legend drawn); a **negative** value means the legend's own colour swatches add ink and hide nothing.

| variant | with legend | legend removed | **data ink hidden** |
|---|---|---|---|
| **shipped figure** | 38 193 | 37 585 | **−608 (hides nothing)** |
| **INSTALLED** (title wrapped 2 lines, label wrapped after "single-exponential") | 36 057 | 35 449 | **−608 (hides nothing — identical to shipped)** |
| rejected: 1-line title, 1-line label | 29 054 | 37 585 | 8 531 = **22.7 %** hidden, title also clipped |
| rejected: 2-line title, 1-line label | 26 544 | 35 449 | 8 905 = **25.1 %** hidden |
| rejected: 2-line title, label wrapped after "estimator" | 33 804 | 35 449 | 1 645 = **4.6 %** hidden |

(The installed variant's absolute ink is 5.6 % below shipped purely because a 2-line title compresses the axes box; the occlusion metric divides that geometry out.)

## 6. Printed-size measurements (scratch build) — footprints, page split, type

Build: `.claude/scratch/v2-figure-text-pass/build/{ctl,new}/`, each a copy of `pj_sub.tex` (md5 `13d3ec2517177b00aead7ba8b2e551cb`, snapshotted 18:35 and **still unchanged in the live folder at install time**) + `neurips_2025_ml4ps.sty` + figs; `ctl` has the shipped PNGs, `new` the installed ones. `/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex` ×3 each.

**Printed image boxes** (`mutool draw -F trace`, pt):

| figure | control | after this pass | Δ |
|---|---|---|---|
| `fig1_gmor` | 396.00416 × **147.1111** | 396.00416 × **147.1111** | **0.00000** |
| `fig_lifetime_headtohead` | 396.02986 × 216.6151 | 396.02986 × 216.6151 | 0 |
| `fig2_anchor_cure_laws` | 396.0121 × 151.20456 | 396.0121 × 151.20456 | 0 |
| `fig3_retention_overlay` | 396 × 257.4 | 396 × 257.4 | 0 |
| `fig3_gmor_condensate` | 396.01899 × 324.01557 | 396.01899 × 324.01557 | 0 |

**Pagination / page split** (`pagesplit.py`, word bboxes vs the 72–720 pt text block) — ⛔ reported, not optimised, per the Head's deferral:

| | control | after |
|---|---|---|
| total pages | **16** | **16** |
| main text ends (References) | **5.46 pp** | **5.46 pp** |
| references block ends (Appendix A) | 7.39 pp | 7.39 pp |
| appendix block | 8.61 pp | 8.61 pp |
| overfull `\hbox` / `\vbox` / underfull | 3 / 0 / 34 | **3 / 0 / 34** (inherited, unchanged) |

**Strongest pagination evidence:** `pdftotext` plain output diff is empty, and a word-by-word comparison of `pdftotext -bbox` gives **8142 vs 8142 words, 0 differing records** — same page, same x, same y, same text for every word in the document.

**Effective type at printed size** (nominal pt × the measured scale `printed_width / canvas_width`; canvas from each PNG's own `pHYs` chunk):

| figure | scale | ticks (≥7) | axis labels (≥8) | titles (≥9) | legend / annot (≥8) |
|---|---|---|---|---|---|
| `fig_lifetime_headtohead` **(changed)** | 1.4707 | 7→**10.29** ✓ | 8→**11.77** ✓ | 9→**13.24** ✓ | 8→**11.77** / 7→**10.29** ✓ |
| `fig3_gmor_condensate` | 1.2501 | 7→8.75 ✓ | 8→10.00 ✓ | 9→11.25 ✓ | 8→10.00 ✓ |
| `fig2_anchor_cure_laws` | 1.1628 | 7→8.14 ✓ | 8→9.30 ✓ | 9→10.47 ✓ | 8→9.30 ✓ |
| `fig3_retention_overlay` | 0.6875 | 10→**6.88** ⚠ | 10→**6.88** ⚠ | 12→**8.25** ⚠ | 8→**5.50** ⚠ |
| `fig1_gmor` **(changed)** | 0.5794 | 9→**5.21** ⚠ | 9→**5.21** ⚠ | 10.8→**6.26** ⚠ | 6.5→**3.77** ⚠ |

⚠ **The two sub-target figures are sub-target in the file that ships today, identically** — both were rendered at their natural canvas (9.49 in and 8.00 in) and are placed at `\linewidth` = 5.5 in, i.e. shrunk 0.58× and 0.69×. **This pass changes none of these numbers**: `fig1_gmor`'s canvas and every font size are unchanged, so its scale is unchanged to 4 decimals. Fixing them means re-rendering at printed size (the `figure-render-pass` recipe) — **out of scope here; flagged in §8.** The one figure I could have degraded (a 1-line title would have taken 0.5794 → 0.5690) was wrapped instead, precisely to avoid it.

## 7. The two `\TODO` tags — discharged by this pass, **NOT removed by me**

| tex line | tag | status |
|---|---|---|
| **l.81** | `\TODO{change "decades" in figure headline to orders of magnitude. do the same for all figures that use decade}` | **DISCHARGED** — 1 hit changed in `fig1_gmor.png`, other four figures swept and clean (§2). Safe to delete. |
| **l.95** | `\TODO{change "published" to the citation number in the plot headline and label, but wait for final draft with .bib added to confirm the number to generate the plot with. also change 5 trained models to "CLU (5 seeds)"}` | **DISCHARGED** — but ⚠ **its literal instruction is superseded.** The Head has since ruled a **descriptive label**, not a citation number; the figure now reads "single-exponential estimator" and **no citation number appears anywhere in it**. Whoever deletes this tag must not re-open it as an owed edit, and the cite pass must not bake a number into this PNG. |

I wrote nothing to `pj_sub.tex`: its md5 is `13d3ec2517177b00aead7ba8b2e551cb` before my snapshot, at install time, and now.

## 8. Reconciliation list (needs an owner)

1. **The unenumerated third "published" site** (§3.2 (d)) — Head/Advisor to ratify or revert (one line, §3.2).
2. **`fig1_gmor.png` and `fig3_retention_overlay.png` are below every type target at printed size** (§6) — inherited from the plain build, which restored banked figures at natural canvas width. `fig1_gmor` is a **main-text** figure at 5.21 pt ticks and a 3.77 pt legend; that is a referee-visible defect. Fix = re-render both at the printed 5.5 in canvas via the `figure-render-pass` recipe (`size.canvas`, 7/8/9 pt rcParams). Owner: a results-analyst figure task. **Not doable inside this pass** — it would change layout, not text.
3. **`pj_sub.tex` l.95's stale instruction** (§7) — owner: whoever deletes the `\TODO`, and the cite pass (must not insert a citation number into any figure).

## 9. Hygiene sweeps

- **Canvas strings** (the only meaningful sweep for a raster figure — every string that reaches the canvas, taken from the executed code): the executed body of `fig1_gmor()` (lines 41–86) and the whole of `final_hth.py` contain **0** hits for `Mo\b|Jawahar|Pierini|published|decade|arXiv|2605.03338|[N]|SF-n|F-n|F5|C2|s42–s46|/Users|scratch|.claude`. Positive control: the same regex fires 1× on `orig_gmor.py` and 2× on `orig_hth.py` (`decade`, `published`). ⚠ `final_gmor.py` is the whole 320-line `make_figures.py` module, whose *other* functions do contain the author token — **but `__main__` calls `fig1_gmor()` only** (verified in source and by the run log printing `FIG1 ONLY DONE` with exactly one PNG written), so none of those strings can reach a canvas.
- **No author name, no citation number, no seed short-tag, no internal ID, no file path** appears in either regenerated image (also confirmed by reading both rendered PNGs).
- **PNG metadata:** the installed `fig1_gmor.png` carries matplotlib's `Software` tEXt chunk (`Matplotlib version3.10.8, https://matplotlib.org/`) — **exactly as the shipped file does**, because `plain_v2_gmor.py` does not pass `metadata={"Software": None}`. Status quo preserved, not introduced. `fig3_retention_overlay.png` (untouched) carries it too; `fig_lifetime_headtohead.png`, `fig2_…`, `fig3_gmor_condensate.png` carry none. Stripping it on the two remaining files is a 1-line change if the Advisor wants uniformity. ⚠ Note for future sweeps: `grep -Eic` over `strings` of a PNG **returns false positives from compressed image bytes** — a 2-letter token like `Mo` fired on all five figures until the itemised re-check showed zero real hits.

## 10. Flag-provenance table

**Zero new measurements were made.** Every plotted number is the banked value re-read from the same `.npz` the shipped figure used; the tap proves it. Inherited provenance:

| figure | source artifact | seeds | non-default flags in effect at the ORIGINAL measurement |
|---|---|---|---|
| `fig1_gmor` | `.claude/outputs/v2-full-runs/gmor_sweep.npz` | designed 42–46 (5) | dim 4, hidden 64, ε=dt=0.05, γ=0.05, 150 epochs; 14 breaking magnitudes δ∈[10⁻⁴,4]; `exact_mode_eigenvalues` map prediction; floor 2ln2/(−ln(1−γ))=27.03; crossover δ*≈0.17 |
| `fig_lifetime_headtohead` | `.claude/outputs/v2-full-runs/gmor_sweep.npz` (same file) | designed 42–46 (5) | as above + lifetime protocol: phase 0.35 rad, threshold 0.2 rad, censoring δ≤3×10⁻⁴, cap 15000 steps; 10/70 runs censored |
| `fig2_anchor_cure_laws` (untouched) | `outputs/v2-referee-experiments/sf3_{gmor,ep}_sweep.npz` | anchored 42–44 (3) | anchored λ=100, 3000 epochs, γ=dt=0.05 |
| `fig3_gmor_condensate` (untouched) | `outputs/f1-gmor-condensate/{gmor_condensate,angular_tilt_contrast}.npz` | 42–46 designed + 42–44 anchored3000 (8 ckpt) | probe-only, float64, linear ambient spurion δ∈[10⁻⁸,0.3] |
| `fig3_retention_overlay` (untouched) | `outputs/v2-prefreeze-baselines/baselines_curves_full.npz` + CLU recompute | coRNN/LEM/LSTM 42–46 (5, median); CLU-emergent 42–44 (3, median); **CLU-designed s42 only (n=1)** | dt=0.05, γ=0.05, 2000 autonomous-hold steps, threshold 0.2 rad, x64 |

**Environment (this pass):** repo HEAD `7fcef501fb1aeae33a8d149b046c9b6126dcecd7`, main venv (`/Users/user/Desktop/CHLU/.venv`), **Python 3.11.13, matplotlib 3.10.8, numpy 2.4.1, jax 0.9.0**, pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026), `mutool` (homebrew), `pdftotext` (poppler, homebrew). No `uv sync` was run; no worktree was created. JAX cold start on `fig1_gmor` (it imports `chlu.experiments.goldstone_harness`): **28.6 s**, not the feared 20 min.

## 11. Files produced

- **Installed (the only two files written outside scratch/outputs):** `.claude/NIPSsubmission/v2-neurreps/figs/fig1_gmor.png` (md5 `eaf697e6f18bb11342ff05b0189ccaa0`, sha1 `16b5c7cf…`) and `.../figs/fig_lifetime_headtohead.png` (md5 `93243884…`, sha1 `26fce076…`). Directory manifest diff before/after install shows **exactly these two lines changed** (`dir_BEFORE_install.txt` / `dir_AFTER_install.txt`, 21 files each).
- `.claude/outputs/v2-figure-text-pass/compare_fig1_gmor.png`, `compare_fig_lifetime_headtohead.png` — before/after sheets rasterised at the true printed width (5.5 in, 300 dpi).
- `.claude/scratch/v2-figure-text-pass/`: `figs-BEFORE/` + `figs-BEFORE.md5`, `figs-AFTER/` + `figs-AFTER.md5`, `gen/{orig,final,alt*,h1..h4,*nl}_{gmor,hth}.py` + `run.py`, `renders/` (all candidates + `TAP_*.json`), `build/{ctl,new}/` (both PDFs + logs), `dir_{BEFORE,AFTER}_install.txt`, `gmor_run.log`.
- **The two installed figures differ from their shipped predecessors by exactly 1 and 4 string literals** — full diffs in §3, reproducible with `diff gen/orig_X.py gen/final_X.py`.

## Git footprint

**None.** `git status --porcelain` empty before and after; HEAD unchanged at `7fcef501fb1aeae33a8d149b046c9b6126dcecd7`. No tracked file created, modified or deleted; everything written is under `.claude/` (gitignored). No branch, no worktree, no commit.

## Open questions / follow-ups / risks

1. **Ratify or revert change (d)** — the third "published" instance (§3.2). This is the only place I exceeded the literal task scope.
2. **`fig1_gmor` is a main-text figure printing 5.21 pt ticks and a 3.77 pt legend.** I could not fix it here (layout ≠ text) but it is the largest legibility defect in the V2 figure set and it is in the main text, not an appendix.
3. **`fig3_retention_overlay`'s designed curve is n=1 (seed 42 only)** while its baseline and emergent curves are medians over 5 and 3 seeds. Not a figure-text issue — but if the caption does not say so, that is a single-seed claim on a shipped figure. (I did not read the caption; the task forbade touching the tex.)
4. **Risk accepted, stated:** `fig2_anchor_cure_laws.png` remains a *reconstruction* (no original generator survives, per `figure-render-pass` §4.2). This pass did not touch it, so that risk is unchanged, not compounded.
5. If the Head prefers the **1-line title** on `fig1_gmor` (panel titles vertically aligned) over the exact footprint, alt A is measured and one re-run away: 144.473 pt tall, 130 words shift ≤1.90 pt, page split still 16 pp / 5.46 pp.

## Proposed handover updates (for the Hub)

- **§1.6 / running log:** *V2 figure text pass (2026-08-24) — done.* `fig1_gmor.png`: "4.5 decades" → "4.5 orders of magnitude" (title wrapped to 3 lines to hold the canvas at 1424×529 px). `fig_lifetime_headtohead.png`: "Published lifetime protocol…" → "Single-exponential estimator lifetime protocol, run on trained CLUs" (2 lines), "published median 1.013" → "single-exponential estimator median 1.013" (2 lines), "5 trained models" → "CLU (5 seeds)", plus a third unenumerated "published" in the in-axes note → "the estimator's own". `fig2_anchor_cure_laws`, `fig3_gmor_condensate`, `fig3_retention_overlay`: **swept, zero hits, byte-identical**.
- **§5 provenance:** the `NIPSsubmission/v2-neurreps/figs/` set is **byte-identical to `papers/plain/v2/figs/`**, not to `papers/neurreps-variants/v2/figs/`. Its two live generators are `scratch/plain-text-builds/plain_v2_gmor.py` (fig1) and `plain_v2f1.py` (head-to-head); both **reproduce the shipped PNGs byte-for-byte today**. `fig3_retention_overlay.png` = banked `outputs/v2-prefreeze-baselines/figA_retention_overlay.png` (generator needs JAX **and checkpoints**).
- **§8 / owed work:** (i) ratify or revert the third-site "published" edit; (ii) **re-render `fig1_gmor.png` (main text) and `fig3_retention_overlay.png` at printed size** — currently 0.58× and 0.69× shrunk, every type target missed, inherited from the plain build; (iii) `pj_sub.tex` l.81 and l.95 `\TODO`s are discharged and can be deleted — but l.95's literal text ("change published to the citation number") is **overruled**; no citation number may enter any figure.
- **Numbers the Hub can quote:** page split unchanged at **16 pp total / 5.46 pp main text**; **all five printed image boxes identical to 0.001 pt**; **8142/8142 word bounding boxes identical** between control and new builds; tap digests `9c284b32…` (fig1, 7 calls) and `baedf798…` (head-to-head, 9 calls) **identical before and after**; both unmodified generators reproduce their shipped PNGs **byte-for-byte**.
- **Standing lesson to record:** *a binary `strings | grep -i` sweep of a PNG produces false positives from compressed image data* — a 2-letter token (`Mo`) fired on all five figures and itemising showed **zero** real hits. Sweep the **generator's canvas strings** plus a visual read; use the binary sweep only for metadata chunks.
- **No code bug for `experiment-engineer`.** Nothing in `chlu/` was touched or found faulty; the only non-obvious behaviour is that `bbox_inches="tight"` makes a title-length change a *footprint* change — handled by wrapping.
