# FINDINGS — pw1-draft-final (the PW-1 closing audit)

Session `pw1-draft-final` · branch `pw1/draft-final` (from `base/pw1-final` 8d3a274) · batmac · start 2026-09-25T12:35:34Z.
The deliverable is **`FINAL-INDEX.md`**, which has jobs 1–6 as tables. The raw material behind it is in `final-125012/`, produced by `pw1_draft_final/run_final.sh`.
Tooling (committed, `pw1_draft_final/`): `compile_tree.sh`, `page_markers.py` (A/B variants), `page_markers_c.py` (the exact variant), `pages.py`, `fig_table_index.py`, `ph_list.py`, `residual_audit.py`, `folder_audit.py`, `hash_placed.sh`, `fix_title.py`, `run_final.sh`, `make_report.py`.
Zero runs, zero fits, zero GPU. Every figure build read only the evidence.

## Status

| job | status | one line |
|---|---|---|
| Step 0 (Hub): in-place `build.sh` in paper-visuals, then README regeneration | **done** | exit 0 and build a = b. **26/26 placed files byte-unchanged in each folder.** The new PROVENANCE.json is cmp-equal to pw1-visuals-tooling's prediction, and the JSON diff is exactly the three edits. README regenerated in both folders |
| 1. Final compile (scratch AND in place), 0 undefined, 0 overfull, style diff EMPTY, page counts equal | **done** | both drafts in both builds: exit 0, 0/0 undefined, 0 errors, **0 overfull boxes** (none to list), 42 = 42 pages; `.aux`/`.bbl`/`.out` and PDF text byte-equal between the scratch and in-place builds; style diff-check ALL EMPTY |
| 1a. Mechanical defect fix | **done (1 fix)** | title line break: the title rendered as "…GENERA-/TION/ON A FROZEN ENCODER". The `\\` moved after "Compositional" and no word changed (below). Installed by the standard method |
| 2. Figure/table index | **done** | 13 figures and 20 tables (plus 1 alias label). Each row comes from the `.aux` `\newlabel` and is cross-checked against the lof/lot entry: **0 disagreements**. Style numbers and pages sit beside them |
| 3. Page counts per section | **done** | measured with markers on a scratch copy; see the method note. Main text **17.92 pp (DRAFT-v2) / 18.09 pp (style) against ICLR's 9**. Total 42 pp |
| 4. Every `\TODO` / `\PLACEHOLDER` by argument | **done** | 4 TODO + 2 PLACEHOLDER per draft, arguments byte-identical in both, with line, section and float |
| 5. Whole-wave residual audit | **done** | 55 rules over two channels (source, PDF text). Every non-zero rule is listed with its reason, and **0 are unclassified**. Three residuals are flagged FOR THE HEAD (below) |
| 6. Figure-folder audit | **done** | 13 stems / 26 files present in both folders with equal hashes. The 4 manifests per folder have 0 problems. Each installed README equals a fresh `make_index.py` regeneration |
| Positive controls | **done** | every scan caught its planted token or tamper (the PDF channel's two known blind spots are listed) |

## Step 0: the figure folders' bookkeeping (before → after)
| file | paper-visuals before → after | w17-paper/figures before → after |
|---|---|---|
| 26 placed figure files | unchanged (sha256 lists cmp-equal: `step0-123639/PLACED-{before,after,final}-*.sha256`) | unchanged |
| PROVENANCE.json | `b2e31817…` → `ec5a6122…` (= `pw1-visuals-tooling/inplace-final-122755/dest2/PROVENANCE.json`) | same |
| SHA256-w22-build{1,2}.txt | `706dc2a4…` → `7a6b64a2…` (headline lines gone, PROVENANCE line new, C-locale order); build1 = build2 | same |
| README.md | `c34ace2e…` → `dd4b960f…` | `ce818d87…` → `64c77b31…` |
| everything else in the tree | unchanged (full-tree hash diff: only the 4 files above) | unchanged |

JSON diff of PROVENANCE.json, old → new: `figures.fig_headline_steps` removed; `fig_headline_steps` removed from `figures_redraw.stems_redrawn_or_added`; `fig_task_cmazet.source_sha256["w21_figures/cmazet_panels.py"]` changed from `59fe2a67…` to `125b40dd…`. That is exactly the three predicted edits.
The README diff is the build commit (f571b2b → f8ef2bb, the last commit that touches the build code; every installed script is cmp-equal to the branch copy) plus page and slot-line numbers. **The old README's pages were stale**: they were written before the wording passes, e.g. fig:crossover p. 26 against the current p. 27, and slot lines 1515 against 1542. make_index read the fresh in-place `.aux`.

## Job 1: the one draft edit (mechanical defect fix)
| | before | after |
|---|---|---|
| source (DRAFT-v2 l.75, style l.81) | `\title{naive-CLU: Deletable Compositional Generation\\ on a Frozen Encoder}` | `\title{naive-CLU: Deletable Compositional\\ Generation on a Frozen Encoder}` |
| rendered title | 3 lines: `NAIVE-CLU: DELETABLE COMPOSITIONAL GENERA-` / `TION` / `ON A FROZEN ENCODER` | 2 lines: `NAIVE-CLU: DELETABLE COMPOSITIONAL` / `GENERATION ON A FROZEN ENCODER` |

Cause: item 13 dropped "Exactly" but left the old `\\`. The small-caps first line then overfilled and LaTeX hyphenated "Generation". Images: `title-DRAFT-v2.png` (before), `stage-title-124855/title-after-*.png` (after).
Checks on stage-title-124855: 0/0 undefined, 0 overfull, 42/42 pp, style diff ALL EMPTY, provenance identical, PH/TODO 2/4 byte-identical, **0 labels moved** (every `\newlabel` equal), and the PDF text differs only in the two title lines.
Install: lock guard clean (live sessions authtest, hubhost, pw1-draft-final); live drafts cmp-equal to `snap-drafts/`; copied; cmp-equal at the destination.
The words are unchanged, so **the recorded title stands: "naive-CLU: Deletable Compositional Generation on a Frozen Encoder"**. The Head may prefer another break.

## Hashes
| file | before this pass (`snap-drafts/`) | after (final, in-place build) |
|---|---|---|
| DRAFT-v2.tex | `8cc405af189adcab3c2974c779a7d0e50bee44686c7ac096fd8078b3f89fa64a` | `ef88256e523603fad7c50b25d2f4d51ba6b39d716c9f8be9be0170529e794f5e` |
| DRAFT-v2-style.tex | `46fbd275c382608fa21b3e0d995f794c07201e9941a9ed259024d988ab38c8c7` | `6f6925f8814dc929415f1bf7a31417faf033d3346df53bfdf7d052f32fbc5a06` |
| DRAFT-v2.pdf | `4e3102e7504ad1ce9c303c7f50ef4a880a5cef4242fc525d82b90f86946b3803` | `a5067a192fe42cd7277499be1d6ac38af74b2e29612bc1a99d1ed3174b049851` |
| DRAFT-v2-style.pdf | `d908315a1f98a976eccf3cde6392268ade1bb1b04bfa0841b7fac96fcec5a702` | `d7f2512ef3125fec3d11b9c50c2c57bde56f2c4a1033ecca5fbcfcd87fadb4fe` |
The in-place compile also refreshed the stale `.aux/.bbl/.blg/.log/.out` in `w17-paper/`, which earlier passes had left from older builds. The old ones are copied in `snap-inplace-before/`. The installed PDFs are the in-place build. They are text- and page-equal to the scratch build, and the bytes differ only through the embedded date and ID. The full list is in `SHA256-final.txt`.

## Measurement notes (choices logged)
1. **pdflatex passes.** From a clean tree the w17 recipe (pdflatex, bibtex, pdflatex ×2) leaves a "Rerun to get cross-references right" warning after the third run. `compile_tree.sh` now runs pdflatex until the warning is gone (at most 5). A clean tree needs 4 runs; the in-place tree needs 2.
2. **Page markers.** The briefed `\typeout{\thepage}` (A) is printed in every row. It lags the shipped page at 17/47 (DRAFT-v2) and 14/47 (style) headings, because it reads the counter before the page is shipped. A `\write` whatsit after each heading (B) gives the shipped page but moved one float in DRAFT-v2 (app:encoder-main p. 36 → 35), so B was rejected. **The page counts use C**: the same whatsit placed inside the heading text. C moves no label, and its page equals the toc page at every numbered heading.
3. **Main text** runs from the title to the first statement heading ("AI use statement"). The AI-use, ethics and reproducibility statements do not count toward ICLR's limit (VENUE.md §1), and neither do references or appendices.

## Positive controls (each scan, planted or tampered on scratch copies)
| scan | plant / tamper | caught |
|---|---|---|
| residual audit, 55 rules | one paragraph with every token, compiled into both drafts (`posctl-124718/`) | src: 55/55 in both drafts. pdf: every rule except `\_` names (pypdf extracts the underscore as a space) and, in DRAFT-v2 only, "expensive" (hyphenated across a page break with Fig. 1's text in between). The src channel covers both. A first run missed a multi-word token split across source lines; fixed (`\s+`) |
| TODO/PLACEHOLDER by argument | 2 `\TODO{}` on one line + 1 `\PLACEHOLDER{}` | arguments TODO 4 → 6, PH 2 → 3; line counts 4 → 5 (a line count misses one) |
| figure/table index vs .aux | lot page of Tab. 4 changed 12 → 13 in a copy of the aux | flagged `tab:del` |
| page markers vs .aux | toc page of Sec. 3 changed 6 → 7 | 1 mismatch flagged |
| folder audit | 1 byte appended to a copied `fig_ladder.png`, and a line appended to the copied README | file unequal; both manifests flag it; README check False; make_index exit 1 |

## For the Head (printed, not changed: none is a mechanical defect)
1. **Length:** main text 17.92 pp (DRAFT-v2) / 18.09 pp (style) against 9, i.e. 8.92 / 9.09 pp over. Figs. 6–8 and 9–13 float in the appendices. MF-11 is held.
2. **"10× steps"**: the fig_ladder legend reads "seed at 10× steps". This follows item 10.6 verbatim, but item 1.8 says never "10× steps" (for the maze figure). The two rulings differ.
3. **WIN / LOSE / PASS / FAIL** in small caps are declared-criterion outcome words: Sec. 3.7 "(−0.1150, S3 LOSE)", and in App. B the Tab. lace-poe caption and its last row. Item 1.3 removed a figure label column, not these.
4. The Fig. 1 image says "corner" ×3 and the Fig. 2 image says "spent/spend" (engine events). Both figures are frozen by items 3 and 6.
5. `tab:cmaze-del` is an unreferenced alias label on Tab. 19. Tab. 19 still carries "Ceiling set by the request design" (a design plateau, not a seed ceiling).
6. The abstract `\TODO` still holds `\S\ref` and `Table~\ref`. It is Head-held and byte-identical.
7. There is one derived number, **106 = 317 − 211**: raw provenance "not found" +1, DERIVED-VERIFIED by `check_derived.py` (DRAFT-v2 l.1009, style l.1004). Reported, not "fixed".

## What did NOT run
- No content edit. The only draft change is the title's line-break position; no word changed.
- No figure was redrawn by hand. The step-0 `build.sh` rebuilt the build figures from the evidence and wrote the same bytes. The mirrored figures (`fig_task_shapes`, `rendered/rc_pw1_*`) were not rebuilt; that needs the dataset, the frozen decoder and model inference.
- No measurement, fit, model load or GPU use.
- The pdf channel of the audit cannot see `\_` identifiers or words hyphenated across a page break with a float in between. For those, the src channel is the audit.
- Figure images (PNG rasters: the rendered Shapes3D panels) contain no extractable text, so their in-image words were not scanned. Their captions were scanned (src).
- The bibliography was not edited. Two cited titles contain "Exact"; they are other authors' titles.
- The reading copy was not mirrored to another machine; no path was named. The batmac paths are `~/clu/outputs/w17-paper/` and `~/clu/outputs/paper-visuals/`.
- The `outputs/pw1-advisor-report.md` was not written: it is the Hub's.
- The memory check before the heavy steps: `vm_stat` showed ~300k free pages (~4.9 GB) before the build; no model was loaded.

## Branch and bundle
Branch `pw1/draft-final` at `513e5ec` (tooling only; no outputs committed). Bundle `pw1-draft-final.bundle` sha256 `cc2c9fba7cbb73e8168d07dbee3952c3d2f4b7988c7687913c6b48dfe5da0ed5`; `git bundle verify`: okay, complete history.
