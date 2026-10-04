# v2-fig1-designed — results-analyst report

Task + acceptance criterion: change exactly one word in one in-image panel title of `figs/fig1_gmor.png` — **"exact on the LEARNED vacuum" → "exact on the DESIGNED vacuum"** — by re-rendering from the generator, with plotted arrays provably unchanged, printed footprint preserved, and the PNG the only file written in `.claude/NIPSsubmission/v2-neurreps/`.

**Status: done, with two reported deviations the Head must see (both anticipated by the task's own method §4 and §3).**

> ⚠ **RECONCILIATION LIST — needs an owner (protocol §5 corollary), 3 items, all in §7:**
> **(1) The task's source pointer is STALE** — `plain-text-builds/plain_v2_gmor.py` no longer generates the shipped PNG (it was superseded by `v2-figure-text-pass`); using it would have silently reverted the "decades" fix. Canonical generator for Fig. 1 is now banked at `.claude/outputs/v2-fig1-designed/final_gmor.py`.
> **(2) The printed box is NOT preserved: −0.927 pt in height** (147.1111 → 146.1843 pt) and no wrap variant recovers it — the shipped 1424 px canvas width is *set by the length of this very title line*. Consequences measured end-to-end (zero reflow, zero pagination change, ≤0.667 pt of caption movement); the alternatives are strictly worse. **Head decision point.**
> **(3) `condensed_paper.tex` (live, edited by the Head at 00:46 during this pass) includes `fig1_gmor.png` TWICE** — at `0.7\linewidth` and `\linewidth`. It inherits the same −0.6 % scale change (measured in §5) and, at `0.7\linewidth`, renders tick type at **3.96 pt**.

**DIAL DECLARATION (echoed): none — a text-on-figure fix. Laundering control: n/a. Falsifies the claim: n/a. Does NOT falsify: n/a. ⛔ ZERO new measurements, ZERO data changes.**
**Pre-registration rule: NOT triggered** — this task measures no ratio, exponent, slope or law. Every number below is a byte/hash equality or a direct geometric measurement from a built PDF, not an estimate. (No `PREREG.md` written; stating this explicitly so the omission is auditable.)

---

## 0. Flag provenance

| item | value |
|---|---|
| repo | `/Users/user/Desktop/CHLU`, HEAD `7fcef50`, branch `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` (pre-existing checkout; **I made no git changes — zero tracked files touched**) |
| interpreter | `/Users/user/Desktop/CHLU/.venv/bin/python` (main venv, per protocol §4 — no worktree, no `uv sync`) |
| generator input data | `.claude/outputs/v2-full-runs/gmor_sweep.npz`, md5 `2ddecfa51fbad753d400e8a65a82d75f` (mtime 2026-07-20; **not regenerated, not touched**) |
| plotted-array seeds/config | inherited from that banked npz: 5 seeds, dim 4, γ = 0.05, dt = 0.05, 14 tilt magnitudes δ ∈ [1e-4, 4], analytic tilts on trained designed checkpoints. **No model was run in this pass; no config flag was in play.** |
| matplotlib rc in generator | `font.size = 9`, `figure.dpi = 150`, `figsize = (9.2, 3.6)`, `savefig(bbox_inches="tight")` |
| LaTeX | `/Library/TeX/texbin/pdflatex`, 3 passes/build; measurement `mutool 1.x` (`/opt/homebrew/bin/mutool`) |
| `pj_sub.tex` snapshot used for the build | md5 `07acd432d56fdc87575516ee9f074772` — **verified unchanged in the live folder before AND after install** |
| `condensed_paper.tex` snapshot used | md5 `af2912caab3b989a53335653139760fe` (the Head's 00:46 revision) |

---

## 1. ⚠ The task's source pointer is stale — and following it literally would have regressed the figure

The task names `.claude/scratch/plain-text-builds/plain_v2_gmor.py` **line 81**, citing `v2-figure-text-pass` as having proved it byte-reproduces the shipped PNG. **That proof has expired**: `v2-figure-text-pass` (2026-08-24, same day) *changed* this figure and installed a new PNG.

| file | title string at line 81 | md5 of the PNG it produces |
|---|---|---|
| `plain-text-builds/plain_v2_gmor.py` (task's pointer) | `…learned vacuum\n(… all seeds, 4.5 **decades**)` | `6e03247a0e5bee5ccad25ccf326315de` — the **pre-text-pass** figure |
| `v2-figure-text-pass/gen/final_gmor.py` (**actual current source**) | `…learned vacuum\n(… all seeds,\n4.5 **orders of magnitude**)` | `eaf697e6f18bb11342ff05b0189ccaa0` = **the PNG shipping today** |

`diff` between the two files is exactly that one line (verified; no other delta). **Editing the task's pointer would have re-installed "4.5 decades" and undone the Head's previous instruction.** I therefore based this pass on `final_gmor.py` and re-proved the provenance from scratch:

`### shipped md5=eaf697e6f18bb11342ff05b0189ccaa0 orig-rerender md5=eaf697e6f18bb11342ff05b0189ccaa0 -> byte-identical=True`

**The canonical Fig. 1 generator is now banked at `.claude/outputs/v2-fig1-designed/final_gmor.py`** (= the installed source, one word changed). `plain_v2_gmor.py` should be treated as historical.

## 2. The change — exactly one word, in exactly one string literal

```diff
--- gen/orig_gmor.py      (reproduces the shipped PNG byte-for-byte)
+++ gen/final_gmor.py     (installed)
@@ line 81 @@ fig1_gmor(), right panel
-    ax.set_title("GMOR spectral-mass law: exact on the learned vacuum\n(max deviation $\\sim 10^{-12}$, all seeds,\n4.5 orders of magnitude)")
+    ax.set_title("GMOR spectral-mass law: exact on the designed vacuum\n(max deviation $\\sim 10^{-12}$, all seeds,\n4.5 orders of magnitude)")
```
Nothing else in the generator, no other panel, no other figure. Visual before/after crop: `.claude/outputs/v2-fig1-designed/title_before_after.png`.

**"learned" elsewhere in this figure: ZERO other occurrences.** `grep -ic learned` on the generator fires **1** (this line). Visual read of the rendered PNG confirms: the left panel title is *"GMOR retention law on **trained** CLUs (5 seeds)"* — no second site; no axis label, tick, legend entry or annotation in either panel carries the word. **Nothing else was changed.**

*Reported, not changed (out of scope, no Head instruction):* `fig3_retention_overlay.png`'s generator (`v2-prefreeze-baselines/make_figures.py` l.74) renders `"S¹ memory retention: **learned** baselines vs CLU"`. This is a **different, correct** sense of the word (trained baseline *models*, not the vacuum), so it is not a designed-vs-learned contradiction. Flagged only so the Hub knows the sweep was done. Zero "learned" in the other three figure generators.

## 3. ⭐ Data-identity proof (the hard acceptance criterion) — PASSED exactly

Instrument: the render pass's data tap (`.claude/outputs/figure-render-pass/tap.py`) — SHA-1 of every numeric array handed to 20 wrapped `Axes` plotting methods, cast `<f8`, compared as a sorted multiset; colours/labels/limits deliberately not recorded; `savefig` redirected so no banked artefact can be overwritten. `__file__` pinned to the original generator path so every relative path resolved as it did when the shipped PNG was made.

Command (one process, both variants):
`PYTHONPATH=/Users/user/Desktop/CHLU /Users/user/Desktop/CHLU/.venv/bin/python .claude/scratch/v2-fig1-designed/run.py`

| variant | data calls | **tap digest** | PNG md5 | canvas px |
|---|---|---|---|---|
| `orig` (= shipped source, unmodified) | 7 | `9c284b32c84f341be86de43696c74e94212b6782` | `eaf697e6f18bb11342ff05b0189ccaa0` | 1424×529 |
| **`new` (INSTALLED)** | 7 | **`9c284b32c84f341be86de43696c74e94212b6782`** | `0a490acb16492e81931be1f5fa72b944` | **1433×529** |

- `### tap digest equal = True` · `### tap multiset equal = True` · `orig-only calls: []` · `new-only calls: []` — **exact equality, not multiset-approximate.**
- **Tap fidelity proof:** the unmodified generator under the tap wrote a PNG **byte-identical to the shipped file** (§1) — the tap is provably faithful to the artefact that ships.
- **Pixel-level corroboration (stronger than the digest).** Left-aligning the two canvases, differing pixels number **6 320** and are confined to **rows 15–37, cols 778–1416** — i.e. *only* the first line of the right-panel title. `diffs in left panel box (rows ≥ 100, cols < 700) = 0`. Every data marker, error bar, axis, tick, legend and the other two title lines are **bit-identical**; the 9 extra px are appended at the right margin.
- Digests: `.claude/outputs/v2-fig1-designed/TAP_gmor_{orig,new}.json`.

## 4. ⚠ Printed footprint — NOT preserved (−0.927 pt), and no variant preserves it

`bbox_inches="tight"` means the canvas width is set by the widest artist, and **that artist is this title line** (figure canvas is 9.2 in × 150 dpi = 1380 px; the shipped bbox is 1424 px, i.e. 44 px of title overhang). One extra character ⇒ 9 px wider ⇒ at fixed `\linewidth` the printed image is 0.63 % shorter.

**Measured from built PDFs (`mutool draw -F trace`, pt) — not `\linewidth` arithmetic:**

| figure in `pj_sub.pdf` | control | after this pass | Δ |
|---|---|---|---|
| **`fig1_gmor`** | 396.00416 × **147.11110** | 395.99638 × **146.18428** | **−0.00778 × −0.92682** |
| `fig_lifetime_headtohead` | 396.02986 × 216.61510 | identical | 0 |
| `fig2_anchor_cure_laws` | 396.01210 × 151.20456 | identical | 0 |
| `fig3_retention_overlay` | 396.00000 × 257.40000 | identical | 0 |
| `fig3_gmor_condensate` | 396.01899 × 324.01557 | identical | 0 |

**I attempted to recover the exact box by re-wrapping (the method the previous pass used successfully) and it is not recoverable.** All four candidates plot identical data (tap digest `9c284b32…` on every one):

| candidate | title layout | canvas px | printed height @ 396 pt | Δ vs shipped |
|---|---|---|---|---|
| shipped ("learned") | 3 lines | 1424×529 | 147.111 pt | — |
| **C1 — INSTALLED: one word, no wrap** | 3 lines | 1433×529 | **146.184 pt** | **−0.927 pt** |
| C2 — break after "law:" | 4 lines | 1395×508 | 144.208 pt | −2.903 pt |
| C3 — break after "on the" | 4 lines | 1395×529 | 150.170 pt | **+3.059 pt (taller — worse)** |
| C4 — break after "designed" | 4 lines | 1395×529 | 150.170 pt | +3.059 pt |

Any wrap short-circuits the overhang entirely and snaps the width to 1395 px (the next-widest artist), overshooting in the opposite direction. **C1 is the minimum-perturbation option by a factor 3 and is the only one that is also literally "one word changed".** C3/C4 would make the figure *taller*, which is the direction that threatens the page limit. ⛔ **No type was shrunk; no `figsize`, `dpi`, `bbox` or `fontsize` was touched** (task method §4 forbids it).

**End-to-end consequence of the −0.927 pt, measured on the full document** (`mutool draw -F stext`, all 1380 text lines, both builds):

- `lines ctl 1380 new 1380` · **`text identical: True`** · **`page assignment identical: True`** · document stays **16 pages**.
- **12 of 1380 lines move.** All on **page 4**, all are the Fig. 1 caption + the `\TODO` box directly beneath the figure. **`max |dy| = 0.66710 pt`, `max |dx| = 0.00000 pt`.** Zero words reflow between lines, zero words change page.
- For scale, the previous pass rejected a variant that moved **130 words by up to 1.90 pt**; this one moves **12 lines by ≤0.67 pt**. It is ~3× smaller than an alternative that was already found to leave pagination unchanged. **Recommendation: accept.** (Reverting is one `cp` — §8.)

## 5. `condensed_paper.tex` — the second, un-scoped consumer of this figure

The live folder contains a newer document, **`condensed_paper.tex` (mtime 00:46, i.e. edited by the Head *during* this pass)**, which includes `fig1_gmor.png` **twice**: `\includegraphics[width=0.7\linewidth]` (p. 3, the main-text slot) and `[width=\linewidth]` (p. 9, App. "Extended Results"). Built control/new in scratch from the 00:46 snapshot:

| inclusion | control box (pt) | new box (pt) | Δ height |
|---|---|---|---|
| p. 3, `0.7\linewidth` | 302.39610 × 112.33675 | 302.40199 × 111.63339 | **−0.70336** |
| p. 9, `\linewidth` | 432.00517 × 160.48506 | 431.99797 × 159.47446 | **−1.01060** |
| all four other figures | — | **byte-identical transforms** | 0 |

Document stays **21 pages**; `lines ctl 976 new 976`, **text identical: True**, **page assignment identical: True**; **28 lines move, all on pp. 3 and 9 (the two captions), `max |dy| = 0.71700 pt`, `max |dx| = 0`.** No reflow anywhere.

## 6. Type targets — FAILED, but inherited, and this pass does not cause it

Effective printed type = `F_pt × (150/72) px/pt × (printed_width_pt / canvas_px)`. Resolved matplotlib sizes for this generator: ticks/axis labels **9.0 pt**, titles **10.8 pt** (`axes.titlesize='large'` = 1.2 × 9), legend **6.5 pt** (explicit).

| inclusion | pt/px | ticks (target ≥ 7) | title (≥ 9) | legend (≥ 8) |
|---|---|---|---|---|
| `pj_sub` `\linewidth` **CTL (shipped today)** | 0.278093 | **5.214** ❌ | **6.257** ❌ | **3.766** ❌ |
| `pj_sub` `\linewidth` **NEW (installed)** | 0.276341 | **5.181** ❌ | **6.218** ❌ | **3.742** ❌ |
| `condensed` p. 3 `0.7\linewidth` CTL | 0.212357 | 3.982 ❌ | 4.778 ❌ | 2.876 ❌ |
| `condensed` p. 3 `0.7\linewidth` **NEW** | 0.211027 | **3.957** ❌ | 4.748 ❌ | 2.858 ❌ |
| `condensed` p. 9 `\linewidth` CTL | 0.303374 | 5.688 ❌ | 6.826 ❌ | 4.108 ❌ |
| `condensed` p. 9 `\linewidth` **NEW** | 0.301464 | 5.652 ❌ | 6.783 ❌ | 4.082 ❌ |

**This is the inherited defect `v2-figure-text-pass` §6 already flagged** — Fig. 1 misses every target by 1.4–4.2 pt *before* this pass. My change moves the numbers by **−0.6 %** (e.g. ticks 5.214 → 5.181 pt), because the same type is now scaled into a 9-px-wider canvas at fixed printed width. It is a rounding error on an existing failure, not a new one, and the only cures are outside this task's scope (raise `font.size` in the generator, or shrink `figsize`). ⚠ **The `0.7\linewidth` inclusion at 3.96 pt ticks is by far the worst site and is in `condensed_paper.tex`'s main text** — the Head should know before that version ships.
**No wrap was forced** (task method §4): the title renders on one line, unclipped, "designed vacuum" fully inside the canvas — verified visually and by the pixel diff bounding box (cols 778–1416 of 1433).

## 7. Reconciliation list (repeat, for the owner)

1. **Retire the `plain_v2_gmor.py` pointer for Fig. 1.** Any future task naming it will silently revert "4.5 orders of magnitude" → "4.5 decades". Canonical source: `.claude/outputs/v2-fig1-designed/final_gmor.py`. → **Hub/curator.**
2. **Head decision on the −0.927 pt box change** (§4). Accepting costs 12 caption lines moving ≤0.67 pt on p. 4 with zero pagination change; rejecting means the figure keeps saying "learned vacuum". No third option exists that is both one word and box-preserving. → **Head.**
3. **`condensed_paper.tex` inherits this figure twice, including at `0.7\linewidth` where ticks print at 3.96 pt** (§5, §6). Also note its Fig. 1 caption and §Results already say "designed" — so before this pass the *image* contradicted its own caption in that document too. → **Head / paper owner.**

## 8. What was written, and how to revert

**Live folder (`.claude/NIPSsubmission/v2-neurreps/`) — exactly one file written:**
```
- MD5 (figs/fig1_gmor.png) = eaf697e6f18bb11342ff05b0189ccaa0   (before)
+ MD5 (figs/fig1_gmor.png) = 0a490acb16492e81931be1f5fa72b944   (after)
```
`diff livefigs-BEFORE.md5 livefigs-AFTER.md5` reports **that single line and nothing else**; the other four PNGs are byte-identical. `pj_sub.tex` md5 `07acd432…` and `pj_sub.pdf` md5 `d0490dfb…` are **unchanged before and after** (re-verified post-install). No `.tex`, `.pdf`, `.aux`, `.log`, and no other figure was touched. All builds were done in `.claude/scratch/v2-fig1-designed/build{,2}/`, never in the live folder.

**Revert (one command):**
`cp .claude/scratch/v2-fig1-designed/figs-BEFORE/fig1_gmor.png .claude/NIPSsubmission/v2-neurreps/figs/fig1_gmor.png`

**Artifacts**
- Backup + manifests: `.claude/scratch/v2-fig1-designed/figs-BEFORE/` + `figs-BEFORE.md5` (md5 **and** sha1); post-state `figs-AFTER/` + `figs-AFTER.md5`; folder-level `livefigs-{BEFORE,AFTER}.md5`, `dir_BEFORE_install.txt`.
- Banked to `.claude/outputs/v2-fig1-designed/`: `final_gmor.py` (canonical generator), `TAP_gmor_{orig,new}.json`, `sweep.json` (the four wrap candidates), the four md5 manifests, `title_before_after.png`.
- Scripts: `.claude/scratch/v2-fig1-designed/{run.py, sweep.py, words.py, words2.py, measure.py}`; renders in `renders/`; builds in `build/` (pj_sub) and `build2/` (condensed_paper).

**Git footprint: none.** No tracked file was created, modified or deleted; no branch, no commit. (`git status --porcelain` shows only the pre-existing state of the `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` checkout, which I did not touch.) ⛔ The author token appears in no figure content, label, filename or artifact produced here.

## 9. Open questions / risks

- **Risk (low, quantified):** the −0.927 pt height change is real. Pagination is unaffected in *both* documents as they stand today, but `pj_sub.tex` is being edited live (its `\TODO` at l.81 is still present and still instructs the now-completed "decades" change — see below); a future edit that lands text near a page boundary could interact. Re-measure if the Head makes large text edits.
- ⚠ **The `\TODO` at `pj_sub.tex` l.81 is now doubly stale**: it instructs the "decades → orders of magnitude" change that `v2-figure-text-pass` already executed. It is *rendered into the built PDF* (it is one of the 12 lines that moved on p. 4), so it will print if not deleted. Not mine to delete — flagged.
- `condensed_paper.tex` l.76's caption and `pj_sub.tex` l.82's caption both read "4.5 orders of magnit**utde**s" / "magnit**utde**" (typo, ×2 in the live tex). Observed while measuring; **not changed** (out of scope, not a figure).
- No code bug for `experiment-engineer` — the generator and the tap both behaved correctly.

---

## Proposed handover updates (for the Hub)

**For §1.6 / figure provenance:**
- **Fig. 1 (`fig1_gmor.png`) source of truth moved.** `plain-text-builds/plain_v2_gmor.py` is **superseded** — it still emits "4.5 decades". The canonical generator is banked at `.claude/outputs/v2-fig1-designed/final_gmor.py` (= `v2-figure-text-pass/gen/final_gmor.py` + the designed-vacuum word). Re-render command: `PYTHONPATH=/Users/user/Desktop/CHLU /Users/user/Desktop/CHLU/.venv/bin/python .claude/scratch/v2-fig1-designed/run.py` (needs `.claude/outputs/v2-full-runs/gmor_sweep.npz`, md5 `2ddecfa5…`).
- **Fig. 1 now reads "exact on the designed vacuum"** in-image, consistent with its caption, §4.1 and the designed-vs-learned honesty architecture. Installed PNG md5 **`0a490acb16492e81931be1f5fa72b944`**, canvas **1433×529**.

**For §5 (provenance) — numbers to fold in:**
- Plotted-data identity: tap digest **`9c284b32c84f341be86de43696c74e94212b6782`** identical before/after (7 data calls, exact multiset equality); pixel diff confined to **rows 15–37, cols 778–1416** (title line only), **0 differing pixels** in the left panel.
- Printed box `pj_sub.pdf`: **396.00416 × 147.11110 → 395.99638 × 146.18428 pt (Δh = −0.92682 pt, −0.63 %)**. Document 16 pp both; **1380/1380 lines keep their page**; **12 lines move, max |dy| = 0.667 pt, max |dx| = 0**.
- Printed box `condensed_paper.pdf`: p. 3 `0.7\linewidth` **112.33675 → 111.63339 pt**; p. 9 `\linewidth` **160.48506 → 159.47446 pt**. 21 pp both; **976/976 lines keep their page**; 28 lines move, max |dy| = 0.717 pt.
- **The exact box is unrecoverable by wrapping** — the shipped 1424 px width is set by this title line; every wrap snaps to 1395 px (−2.90 or +3.06 pt). C1 is the minimum-perturbation option.

**For §8 (open issues / known defects):**
- **Fig. 1 type is below every target at printed size, inherited:** ticks **5.18 pt** (target ≥ 7), titles **6.22 pt** (≥ 9), legend **3.74 pt** (≥ 8) at `\linewidth` in `pj_sub`; **3.96 / 4.75 / 2.86 pt** at `0.7\linewidth` in `condensed_paper` p. 3. Pre-existing (5.21 / 6.26 / 3.77 pt before this pass); this pass moved it by −0.6 %. Cure requires raising `font.size` or shrinking `figsize` in the generator — a scoped task, not done here.
- **`pj_sub.tex` l.81 `\TODO` is stale and renders into the PDF** (instructs an already-completed change).
- **"magnitutde" typo ×2** in the live `pj_sub.tex` (l.82) and `condensed_paper.tex` (ll.76, 176) captions.
- `fig3_retention_overlay.png` legitimately renders the word "learned" ("learned baselines vs CLU") — a *different* sense; **not** a designed-vs-learned contradiction, deliberately left alone.
