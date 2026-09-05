# v1-figure-fix-and-insert — results-analyst report

**Task + acceptance criterion:** fix three presentation defects in `fig1_certificate.png`,
`fig_frontier_clean.png`, `fig4_bibo.png` without moving any plotted value, then insert
exactly two `figure` environments into `pj_sub.tex`.
**Status: done.**

**⚠ FIRST 10 LINES — two items that need an owner (protocol §5 corollary):**
1. **A1 was already fixed before I was spawned.** `fig1_certificate.png` already reads
   `squeeze: beyond the swept ζ ≤ 2.0`; `priced` = 0 in it. The task premise was stale.
   **I changed nothing in fig1.** Someone should confirm no other task file is still
   carrying the "fig1 says priced" premise.
2. **`.claude/NIPSsubmission/v1-ttcl/pj_sub.pdf` is stale and was already stale on arrival**
   — 14 pages, embeds the *pre-replot* 2480 px fig1. It predates the replot pass, not just
   this one. Outside my write scope, so untouched. **Needs an owner to refresh** or a
   reviewer opening the directory reads a two-passes-stale paper.

**DIAL DECLARATION (echoed from the task).**
**Dials touched: NONE.** Presentation-only re-renders from banked data plus two figure
insertions. No experiment, no configuration change, and — proven by data tap — no plotted
value moved. *Laundering control:* the data tap itself; re-running the unmodified staged
generators must reproduce the banked PNGs byte-for-byte before any "unchanged" claim counts.
*Falsifies:* any digest mismatch on any of the three figures. *Does NOT falsify:* a changed
figure **footprint** — moving a legend or a canvas necessarily moves the printed box, and
that is the fix working, not a regression.

**Deliverable #1: `.claude/NIPSsubmission/v1-ttcl/BUILD-NOTE-R4.md`** (340 lines) — full
string-by-string log, digest table, printed-box table, both insertions quoted, sweep tables.
This file is the deliverable; below is the condensed account.

---

## What I did

- **Pin check first: PASSED.** `pj_sub.tex` md5 `08d31733b5648ed6ab4a6bbc5dc07ed8`, 382
  lines — matched the task pin exactly. Did not abort.
- Re-ran all three *staged* generators from `.claude/scratch/v1-replot-pass/` under `tap.py`
  to establish that the baseline is reproducible, not merely asserted. All three reproduced
  the banked PNGs **byte-for-byte** and all three digests matched the staged baseline.
- **A1 `fig1_certificate.png`: no change made, none needed** (§Findings 1).
- **A2 `fig_frontier_clean.png`:** removed the two-row `500`-tick hack; un-wrapped both
  y-labels to single lines; equalised the two title boxes; re-targeted the canvas to the
  *actual* printed width `0.82\linewidth` = 5.330 in.
- **A3 `fig4_bibo.png`:** moved the legend from inside the axes (`loc="center left"`) to
  outside, below the x-label. One line changed; canvas grew in height to hold it.
- Verified digests unmoved for all three, then installed the two new PNGs.
- Inserted the two figure environments into `pj_sub.tex` via an assertion-guarded script,
  stripping the `⛔` glyph from the C2 caption as mandated.
- Built before / after and measured every printed box from the PDF with `mutool trace`.

## How I verified

```
# baseline reproducibility (staged generators, unmodified)
cd .claude/scratch/v1-replot-pass
../../../.venv/bin/python ../v1-figure-fix/runtap.py base_fig1 new_fig1.py
  -> [tap] 13 data calls, digest 88273eb233a356633d565825aacd7351fd9f9343
  -> md5 of re-render == banked figs/fig1_certificate.png == 47adc87bf09ccef4fbf08df504584677
# ... fig4 -> ae96d8b3…, 5fdd6bd2…   frontier -> e02915ed…, 2a217098…   (all byte-identical)

# fixed figures
cd ../v1-figure-fix
FIG4_NCOL=1 FIG4_H=3.84  ../../../.venv/bin/python runtap.py fix_fig4     fix_fig4.py
FRONTIER_VARIANT=pad     ../../../.venv/bin/python runtap.py fix_frontier fix_frontier.py

# build
/Library/TeX/texbin/pdflatex -interaction=nonstopmode -halt-on-error pj_sub.tex   (×2)
mutool trace pj_sub.pdf | /usr/bin/grep fill_image
../v1-replot-pass/pagesplit.py pj_sub.pdf "MAIN=Flag-Provenance" "APPX=References"
```

Environment: `.venv` Python, **matplotlib 3.10.8, numpy 2.4.1**, Agg, dpi 400; `pdflatex`
TeX Live `/Library/TeX/texbin`; `mutool 1.27.2`; poppler `pdftotext`. **No JAX imported, no
`chlu/` code touched, no experiment run** — all three figures are re-renders from banked JSON
(`paid_access_metrics.json`, `regime-remap-2000ep/runs/*.json`).

## Findings

### 1. ⛔ A1 was already correct — the task premise was stale
| check | observed |
|---|---|
| annotation as actually drawn (live `Text` artist) | `squeeze: beyond\nthe swept $\zeta \leq 2.0$` |
| `priced` in the 81 text artists drawn into fig1 | **0** (positive control `squeeze` = 3) |
| `figs/fig1_certificate.png` md5, before = after | `47adc87bf09ccef4fbf08df504584677` |

Timestamps explain it: the prior spoke's `new_fig1.py` is stamped **20:47:45** and the banked
PNG **20:48:19**, whereas fig4 and the frontier are stamped **20:33:11**. The corrected string
landed ~15 min after the rest of that pass — after the Advisor had read the figure. **Nothing
in fig1 was touched.**

### 2. A2 — the detached `500` had a specific cause, and the compensation was unnecessary
The staged generator ends with `a.get_xticklabels()[0].set_y(-0.095); a.xaxis.labelpad = 7.0`
— an explicit "drop the crowded 500 onto a second row" hack. **Both lines removed.** The
crowding does not exist at the true printed width:

| tick pair | measured gap (printed pt) |
|---|---|
| 500 → 1000 | **+4.33** (positive = clear) |
| 1000 → 2000 | +22.02 |
| 2000 → 4000 | +61.88 |

y-spread of the four tick-label boxes: **0.000 px** on both panels (was ≈ 9.6 pt for label 0).
Y-labels un-wrapped to `CLU-EBM storage fidelity` (98.8 pt) and `CLU gated accuracy`
(79.9 pt), both inside the 105.4 pt axes height — **no word changed**, only the line break.
Symmetry: a one-line left title measures **184.0 pt** against a **152.4 pt** panel and cannot
fit, and `3 seeds` must stay, so the *right* title got an **empty second line** instead
(`"Gated acc vs epochs" + "\n "`). Both title boxes are now 18.6 pt and both titles are
top-aligned. The right title's **text is unchanged**.

### 3. ⛔ A3 — the fig4 legend was covering the caption's own evidence
Measured against the unmodified staged generator: the inside legend covered **24.3 % of the
axes area** (325 666 px²) and hid **41.6 % / 41.6 % / 54.7 %** of the three arms' plotted
lengths plus 32.9 % of the coercive-edge line. That matters beyond aesthetics: b = 1.0/2.0/3.0
is exactly where all three arms **coincide** (r\* = 1.0114 / 2.0079 / 3.0125, identical across
arms), which is what the C1 caption's "the state-replacing map coincides exactly with the
screen-ignored ablation" asserts. After the move: **0 px² overlap, 0/4000 points occluded on
every curve, 0/2 endpoints on both reference lines** — enforced by a hard assertion that
aborts the render. Data area preserved: axes 273.5 × 158.7 → **273.6 × 159.2 pt**.

Rejected alternatives, with the measurement that killed each: two-column legend below =
331.1 pt wide vs a 318.2 pt canvas (`unclip` → `STILL CLIPPED`); legend kept inside = 202.1 pt
= 73.9 % of axes width against data-free bands only 41 pt tall vs a 52.2 pt legend, so it
needs either an occluded curve or ~1.8 padded empty decades on the log axis.

### 4. ⛔ Data tap — no plotted value moved
| figure | staged baseline | this pass | calls | verdict |
|---|---|---|---|---|
| fig1 | `88273eb233a356633d565825aacd7351fd9f9343` | same | 13 = 13 | **MATCH** |
| fig4 | `ae96d8b34e077002514d2a36e7a14e08c46329f9` | same | 5 = 5 | **MATCH** |
| frontier | `e02915edfab102311999edffb248c2908ad55fe2` | same | 8 = 8 | **MATCH** |

Full sorted call lists compared element-wise, not just the sha1.

### 5. A third frontier defect, not in the task: type would have printed 1.14× oversize
The staged frontier PNG was built for a **4.680 in** box, but C2 specifies
`width=0.82\linewidth` = **5.330 in**. Shipping it as-is would have upscaled its type by
**1.1389×** → titles 10.25 pt, labels 9.11 pt, ticks 7.97 pt: legible, but visibly larger than
fig1 and fig4 and off the house 9/8/7. Canvas re-targeted to 5.330 in exactly.

| figure | scale before | scale after | title/label/tick after |
|---|---|---|---|
| fig1 | 0.998470 | 0.998470 (unchanged) | 9.006 / 8.008 / 7.009 ✅ |
| fig4 | 1.000550 | **1.000060** | 9.001 / 8.000 / 7.000 ✅ |
| frontier | **1.138900** ⚠ | **1.000030** | 9.000 / 8.000 / 7.000 ✅ |

### 6. Printed boxes, measured from the PDF (`mutool trace`, never `\linewidth` arithmetic)
`\linewidth` = **468.00288 pt**. "Before" for fig4/frontier is a real build of the inserted
tex carrying the *old* PNGs, so it is like-for-like at the same width.

| figure | before (pt) | after (pt) | ΔW | ΔH |
|---|---|---|---|---|
| fig1 (`\textwidth`) | 468.003 × 169.840 | 468.003 × 169.840 | 0.000 | 0.000 |
| fig4 (`0.68\linewidth`) | 318.235 × 212.157 | 318.259 × 276.497 | +0.024 | **+64.340** |
| frontier (`0.82\linewidth`) | 383.764 × 147.601 | 383.772 × 158.405 | +0.008 | **+10.803** |

### 7. Insertions and build
`diff` on `pj_sub.tex`: **exactly 2 hunks, 20 inserted lines, 0 deleted, 0 modified.** Both
anchors matched exactly once. Captions are the Advisor's draft verbatim; the `⛔` before
"Note this panel is $n=3$" stripped to plain prose; glyph count in inserted text = 0.

| build | errors | undef | warnings | pages | main | appendix | refs |
|---|---|---|---|---|---|---|---|
| before insertion | 0 | 0 | — | **14** | 8.44 pp | 4.15 pp | 1.41 pp |
| inserted, old figs | 0 | 0 | — | 15 | 8.44 pp | 5.11 pp | 1.45 pp |
| **inserted, new figs (final)** | **0** | **0** | **0** | **15** | **8.44 pp** | **5.23 pp** | **1.33 pp** |

Insertions cost **+0.96 pp** of appendix; the re-renders a further **+0.12 pp**.
**Main text is 8.44 pp in all three builds — unchanged.** Fig. 2 (BIBO) lands at the top of
**p. 11**, Fig. 3 (frontier) at the top of **p. 13**, each on its own subsection's page.

### 8. Residual sweep, positive-controlled
PNGs carry no text layer, so each generator was re-executed and **every** `Text` artist
actually drawn was harvested and searched.

| target | artists | `priced` | `receipt` | `ledger` | `no-physics router` | positive control |
|---|---|---|---|---|---|---|
| fig1 | 81 | 0 | 0 | 0 | 0 | `squeeze` = 3 ✅ |
| fig4 | 47 | 0 | 0 | 0 | 0 | `screen` = 3 ✅ |
| frontier | 58 | 0 | 0 | 0 | 0 | `epochs` = 4 ✅ |
| the two new captions | — | 0 | 0 | 0 | 0 | `wormhole` = 1 ✅ |
| whole `pj_sub.tex` | — | 0 | 0 | 0 | 0 | `wormhole` = 21 ✅ |

All via `/usr/bin/grep -o … | wc -l` or Python substring counts (never `grep -c`); every
negative carries a positive control on the same corpus.

## Acceptance criteria

| criterion | result |
|---|---|
| Pin check passed, or aborted | **PASSED** — `08d31733…`, 382 lines |
| `diff` shows exactly two hunks, both figure environments, zero other changes | **2 hunks, 0 deletions, 0 modifications** |
| `submission.tex` and `.claude/papers/v1-short/**` byte-untouched (md5 manifest) | **verified before and after**; `submission.tex` = `caef2272f9dc96d349b46486563d24ee`, 11-file manifest in BUILD-NOTE-R4 §0 |
| `tap.py` digests match the staged baseline for all three | **3 / 3 MATCH** |
| `priced` appears in no PNG | **0 in all three**, positive-controlled |

## Git footprint
**None.** `git status --porcelain` = **0 lines** before and after. Every write is under
gitignored `.claude/`: `NIPSsubmission/v1-ttcl/{pj_sub.tex, BUILD-NOTE-R4.md, figs/*.png}`,
`scratch/v1-figure-fix/**`, `outputs/v1-figure-fix-and-insert.md`. No branch, no commit.
Rollback: `.claude/scratch/v1-figure-fix/build/pj_sub.tex.PRE` and
`.claude/scratch/v1-figure-fix/backup_figs_pre/` hold the pre-pass tex and both old PNGs.

## Open questions / follow-ups / risks

1. **`pj_sub.pdf` refresh needs an owner** (see the top-of-report flag). Current 15-page
   build sits at `.claude/scratch/v1-figure-fix/build/after/pj_sub.pdf`.
2. **fig4 is now a tall figure** — 318.3 × 276.5 pt, ~0.53 of a page with its caption. That
   is the price of an unoccluded 202 pt legend at a 4.42 in width. If the appendix budget
   tightens, the lever is `width=0.60\linewidth` (re-render required to hold 9/8/7 pt), not
   putting the legend back inside.
3. **Frontier placement (report-only, task §C):** it could go to §4.1 main text. Measured
   cost ≈ **+0.35 pp** onto a main text already at **8.44 pp** against a ~5 pp target.
   Recommend it stays in the appendix. **The Head decides.**
4. **Five caption edits I believe are needed but did **not** apply** — enumerated in
   BUILD-NOTE-R4 §9. The load-bearing one: C2 says "kv$32$ over-trains between $2000$ and
   $4000$ epochs", which the **gated-accuracy** panel supports (1.00 → 0.89) but the
   **fidelity** panel does not (kv32 ends at 0.99). Suggest "over-trains in gated accuracy".
5. **No code bug for `experiment-engineer`.** The `500`-tick hack and the frontier's
   4.68 in canvas target are both in gitignored scratch, not in `chlu/`.

## Proposed handover updates (for the Hub)

**§1.6 / experiments — nothing to add.** No dial was exercised; no experimental number
changed. The only new *measurements* are presentation metrology, below.

**§5 provenance — V1 short (`v1-ttcl`) figure state, as of this pass:**
- `pj_sub.tex` md5 **`727ebee2b8498b4095f8bb7159258f90`**, **402 lines** (was
  `08d31733b5648ed6ab4a6bbc5dc07ed8` / 382). Two `figure` environments added; nothing else.
- Figures now in the V1 short: **3** — `fig1_certificate` (§2, `\textwidth`, `fig:reach`),
  `fig4_bibo` (App. B.2, `0.68\linewidth`, `fig:bibo`, p. 11),
  `fig_frontier_clean` (App. C.3, `0.82\linewidth`, `fig:frontier`, p. 13).
- Banked PNG md5s: fig1 `47adc87bf09ccef4fbf08df504584677` (**unchanged**),
  fig4 `92059f182d341ed96d2f15c39ec0c081` (was `5fdd6bd2f2dcf06a4615cc491023ac7d`),
  frontier `35479c5e95100db96ad25d70fb0a5f2a` (was `2a217098434d80a15e98a12419109182`).
- **Data-tap digests are the invariant across all V1 replots** and must be quoted in any
  future figure task: fig1 `88273eb233a356633d565825aacd7351fd9f9343` (13 calls),
  fig4 `ae96d8b34e077002514d2a36e7a14e08c46329f9` (5), frontier
  `e02915edfab102311999edffb248c2908ad55fe2` (8).
- Build: **15 pages**, 0 errors / 0 undefined refs / 0 LaTeX warnings. Split:
  **main text 8.44 pp** (unchanged by this pass), appendix 5.23 pp, references 1.33 pp.

**§8 / running log — three items worth folding in:**
1. **The "fig1 says `priced`" premise is retired.** It was already false when scoped; fig1 is
   byte-unchanged at `47adc87b…` and `priced` = 0 across all 81 of its drawn text artists.
   The document-wide `priced` count is **0** in `pj_sub.tex` too.
2. **New standing rule candidate — figures must be built for the width they are inserted at.**
   The frontier was banked for 4.680 in but C2 inserts it at 5.330 in: a silent **1.1389×**
   type upscale that no `\linewidth` arithmetic would have caught, only a `mutool` measurement
   of the built PDF. Recommend the Hub require, in every future figure task, that the
   `\includegraphics` width fraction be named **at scoping** so the canvas is targeted to it.
3. **Occlusion is measurable and should be a standing check.** The fig4 legend hid **54.7 %**
   of one arm and **24.3 %** of the axes; the harness that measures this
   (`fix_fig4.py`, sampled-point-in-legend-bbox, asserting 0) is ~25 lines and is reusable.
   Suggest it join `tap.py` as a required figure gate: *values unmoved* **and** *values visible*.

**Main-text budget (unchanged fact, now measured):** main text is **8.44 pp** against the
~5 pp target. Neither insertion touched it — both are appendix floats.
