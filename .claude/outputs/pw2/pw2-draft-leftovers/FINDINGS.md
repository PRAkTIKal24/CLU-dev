# FINDINGS — pw2-draft-leftovers (PW-2 item 4: the leftovers, BOTH drafts, the ladder figure, and the item-5 follow-up on the limitations list)

Session `pw2-draft-leftovers`, branch `pw2/draft-leftovers` (from `pw2/draft-tells` 5a3d1a2), batmac.
- Snapshot: `snap-start/`, with sha256 in `snap-start/SHA256`. It equals pw2-draft-tells' `INSTALLED-SHA256.txt`.
- Baseline build (unedited copy of the live folders): `base-20260925T173204Z/`.
- Stage: `stage-20260925T173204Z/`.
- Figure builds: `fig-20260925T172427Z/`.
- Install verification: `verify-live-20260925T173629Z/`.

## Status: all seven parts DONE in both drafts and installed. Acceptance met.

| brief part | result |
|---|---|
| 1. Ladder legend | **Done.** The legend string is now "seed at $10\times$ training steps".<br>• **Reproduces the installed file:** the paper-visuals pipeline (`build.sh`, folder layout), run unchanged on a copy of `paper-visuals/`, rebuilds the installed `fig_ladder.pdf` exactly (949d5b6d…, two builds).<br>• **One-string edit:** in `make_figures_w22.py` (the same file in `w21_figures/`, committed): `r"seed at $10{\times}$ steps"` → `r"seed at $10{\times}$ training steps"`.<br>• **New build:** two builds are byte-identical (`REBUILD: BYTE-IDENTICAL`). `fig_ladder.pdf` is 5211de78…<br>• **Nothing else changed:** only `fig_ladder.{pdf,png}` and `PROVENANCE.json` differ between the build manifests. Every other figure is byte-equal.<br>• **Provenance:** its JSON diff is one leaf, `figures/fig_ladder/rendered_text[23]`, the legend string. Every plotted value, source and seed record is unchanged.<br>• **Rendered text:** the diff (`fig-*/LADDER-TEXT-DIFF.txt`, pypdf) is one line, `seed at 10× steps` → `seed at 10× training steps`. The PNG was checked by eye: the legend is at the right and not clipped.<br>• **Install:** only `fig_ladder.pdf` was installed, into `~/clu/pw2/live/{v2,style}/figures/`. It is cmp-equal to the build (5211de78… in both). The old file is in `snap-start/*/figures/`. |
| 2. Figs. 5–8 cited from their result sentence | **Done by citing. No float moved.** One `Fig.~\ref` is inserted into each existing sentence that states the result:<br>• Fig. 5 `fig:ladder`, Results/ladder: "On every rung and every seed the n-CLU's score in Tab. 3 exceeds every baseline's printed score (Fig. 5)."<br>• Fig. 6 `fig:crossover`, App. K-curve: "… is what the rule returns, not a K that three seeds distinguish from K=24 or K=28 (Fig. 6)." See the note for the Head below.<br>• Fig. 7 `fig:deletion`, two sentences: Results/deletion "… by about six tuples per held-out combination, on all three seeds (Tab. 4, Fig. 7)." (panel a); Results/retrained baselines "… against the LSTM at seed 2 (Fig. 7):" (panel b).<br>• Fig. 8 `fig:adapt`, Results/two ablations/encoder: "… on two of three under a low-rank adapter (Tab. 15, Fig. 8)."<br>`\ref` counts per label, 0 → 1, 1, 2, 1, in both drafts (`CHECKS.md`). |
| 3. The byte-ratio `\TODO` sentence | **Dropped, with a `% PW2-4` pointer** that keeps its full text. `\TODO` **4 → 3** in each draft. **Removed:** "no record prints the byte ratio of the composition-only n-CLU to the single design; …". **Left:** (1) the abstract's maze sentence (Head-held); (2) the AI use statement; (3) the anonymised code archive. |
| 4. The two undrawable observation-only `\PLACEHOLDER`s | **Cut.** `\PLACEHOLDER` **2 → 0** in each draft. Removed:<br>• in the `fig:cmazet-families` caption, "per-model $s_{t+1}$ for the observation-only input of Tab. cmazet …", with its lead-in "The models of Tab. cmazet were not persisted:";<br>• in the `fig:cmazet-door-tint` caption, "per-model $s_{t+1}$ on the door-then-tint transitions for the observation-only input …", with its lead-in "For the models of Tab. cmazet:".<br>**One sentence added** in App. mazewm (Maze world-model rows), directly after Tab. cmazet: "The observation-only models of Tab.~\ref{tab:cmazet} were not persisted, so their per-model panels are not drawn."<br>Two consequential cuts, purely subtractive, each with a `% PW2-4` pointer. Both statements would have been false at 0 placeholders:<br>• the main-text sentence "The rows still pending a run are marked placeholders whose text names the row." (end of Introduction);<br>• in App. Placeholder index: the two sentences on the two remaining placeholders and the clause "their two placeholders name the per-arm outputs …". |
| 5. Machine paths in `.tex` comments | **Done: 2 → 0 absolute paths per draft.** The scan covered the whole file, comments included: `/Users/`, `/home/`, `C:\`, `~/`, `/c/chlu`, `/tmp/`, `/private/`, `/Volumes/`. List below. The provenance checker still resolves every SRC path: its findings are identical. The SRC paths were already relative to the evidence root, and the two home-directory paths were descriptive comments only. |
| 6. The 9-pt quote block | **Done.** `\begin{quote}\small` → `\begin{quote}`. Measured in the PDF (pypdf text runs, `QUOTE-SIZE.txt`): the quote is set at 8.97 pt before and **9.96 pt after**, the body size, in both drafts. |
| 7. Item 5 on the Discussion limitations list | **Done, in lock-step** (the list is byte-identical between the drafts before and after).<br>• **Digit tokens in the list: 27 → 10.** In the whole Discussion, 33 → 16. The 10 left are identifiers: seed 0 ×2; "random 1%"; $K=24$; the tokeniser's "6" inside $K\ge36$; the "4" inside float64; rank 64; row 3; seeds 3 and 4.<br>• **Every number that left** lives in a table or caption the text now cites, or is Head-held elsewhere, and each keeps a `% PW2-5 moved` pointer with SRC.<br>• **ADDED to a caption:** one sentence to the Tab. kcurve caption (K=40, 7 atoms per well, 0.991 times the reference, validation composition 0.0648), because no float carried it. The provenance checker finds every added number in its SRC.<br>• **Rounded values:** ROUNDED 9 rows, **0 FAIL**. Positive control: a planted "about four times" against 3.9942 FAILs.<br>• **524,288 left the list.** It was not the reference definition; that is in Methods. The list now says "the reference".<br>• **NUMBERS-LEFT:** 0 UNCLASSIFIED. The Discussion rows are in `NUMBERS-LEFT-DISCUSSION.md`. |
| Head-held text | **Byte-identical.** `headheld.py` PASS 7/7 in both drafts, on the stage and on the installed files, against pw2-draft-tells' `HEAD-HELD.md`. It is copied here unchanged, since no Head-held paragraph was touched. |
| Compile (COMMON) | Both drafts: exit 0; 0 undefined citations; 0 undefined references; 0 errors; 0 multiply defined; **0 overfull**. Underfull (all `\vbox … while \output is active`): v2 18 → 16, style 18 → 20. 56 bibitems. No Rerun. **Pages:** v2 42 → 42, style 44 → 43 (`COMPILE.md`). |
| Style diff-check | **ALL DIFFS EMPTY.** Number multiset 3,362 = 3,362 (1,758 distinct); it was 3,377 before this pass. Placeholders, artefact paths (607 = 607), SRC strings and heading sequence are all equal. |
| Provenance | **Identical** in both drafts, line numbers masked (`PROVENANCE-COMPARE.txt`). v2: 3 unsourced lines, 9 not-found tokens. Style: 3 and 10. All pre-existing. Checked tokens 2,355 → 2,342. |
| Labels | 77 → 77 in each draft; 0 missing, 0 new, 0 renumbered. Pages moved:<br>• v2: `sec:limits` p.17→16, `app:laws` 39→38, `app:notrun` 42→41;<br>• style: `app:encloop` 31→32, `app:clue` 32→33, `app:notrun` 43→42. |
| Installed | `.tex` and PDF in `~/clu/pw2/live/{v2,style}`, cmp-equal to the stage, plus `fig_ladder.pdf` in both `figures/`. The lock guard was clean, and the live files were cmp-equal to the snapshot just before install. A fresh build from a copy of the live folders gives identical `\newlabel`s and the same metrics. sha256 in `INSTALLED-SHA256.txt`. |
| Bib | `refs_clu_iclr.bib` is b2cf715f… in both folders at the start and at the end. It was never written. |

## Part 5: absolute paths, before → after (same edit in both drafts)
| draft line (v2 / style) | before | after |
|---|---|---|
| 12 / 18 | `%     (called EV below; on the drafting machine it is ~/clu/evidence/outputs/).` | `%     (called EV below: the outputs/ directory of the evidence tree).` |
| 35 / 41 | `% COMPILE:  pdflatex DRAFT-v2.tex (from ~/clu/outputs/w17-paper/, template in ./iclr2027/)` | `% COMPILE:  pdflatex DRAFT-v2.tex (from the folder that holds this file, template in ./iclr2027/)` (style: `DRAFT-v2-style.tex`) |

The final grep over both installed `.tex` files for `/Users/|/home/|C:\|~/|/c/chlu|/tmp/|/private/|/Volumes/` finds **0** (`CHECKS.md`). Relative SRC paths of the form `../../outputs/…` stay: they resolve from the evidence root, as the provenance checker reads them.

## Part 7: the limitations list, before → after (numbers only; full text in `LIMITATIONS.md`)
| item | before | after | where the number lives now |
|---|---|---|---|
| 2 conjunction deletion | "by 6.13 / 6.43 / 6.30 tuples …, 32.1 on the combinations that request red with another shape at seed 0, while a random 1\% deletion moves them by at most 0.11" | "by about six tuples …, about thirty on the combinations that request red with another shape at seed 0, while a random 1\% deletion moves them by at most about a tenth of a tuple" + `Tab.~\ref{tab:del}` | Tab. del column (d); Tab. del caption notes (−32.1); Fig. deletion caption (6.13, 6.43, 6.30; 0.02, 0.11, 0.01) |
| 3 address-book refit | "0.97 GB, … its refit takes 2,367 to 5,913 s, longer than a baseline's retrain-without" | "nearly a gigabyte, … its refit takes longer than a baseline's retrain-without (Tabs. del and rivals-del)" | Tab. del cost rows (967,304,800 B; 2,367.3 … 5,912.9 s); Tab. rivals-del |
| 4 bytes | "4.0768 to 4.0897 times the reference"; "minimum 3.9942 times the reference at $K=24$"; "($K=40$, 7 atoms per well, 0.991 times the pin) composition collapses to 0.0648"; "inside 524,288 bytes" | "about four times the reference"; "minimum at $K=24$ (Tab. kcurve)"; "composition collapses (Tab. kcurve, caption)"; "inside the reference" | Tab. ledger-kstar; Tab. kcurve (3.9942); **Tab. kcurve caption, ADDED** (K=40, 7 atoms, 0.991, 0.0648; SRC `w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md`); 524,288 = the Methods definition |
| 5 truncated search | "the MLP at seed 0 still gains 1.47 tuples" | "… still gains more than a tuple … (Sec. 4.1, Tab. t7)" | Results first paragraph (Head-held, "+1.47"); Tab. t7 diagnostic MLP row |

Items 1, 6, 7, 8, 9 and 10 carried no table-readable number and were not touched.

**Record check.** Every value was checked verbatim against its record before it was written (`ROUNDED.md`). One correction came out of the check: 4.0768 is not in `w13-seeds-and-tables/TABLES.md`, which prints only the seed-0 4.0897. The per-seed 4.0897 / 4.0768 / 4.0847 are in `w13-referee-2/REFEREE-v1.md`, which is what Tab. ledger-kstar cites. The ROUNDED row and the moved-number record now cite that file. The `.tex` pointer lists both files.

## For the Head (reported, not edited)
1. **Fig. 6: the main-text result sentence is Head-held.** The only main-text statement of the K-curve result (the plateau and the footprint minimum near K=24) is the Head-held Experiments first paragraph. Fig. 6 is therefore cited from App. K-curve's plateau sentence, next to the float. The Head may add `Fig.~\ref{fig:crossover}` to the Experiments paragraph.
2. **App. "Placeholder index" now indexes no placeholder.** It keeps its list of rows that were pending earlier and are now written in. The heading is the Head's; appendix headings are not changed in PW-2. The section can be dropped or renamed in the Head's pass.
3. **The preamble comments still describe the `\PLACEHOLDER` convention** (lines 15–19, 60–64 in v2). They are comments only, and the macro definition stays harmless with 0 uses.
4. **The ladder legend lives in two copies.** `~/clu/outputs/paper-visuals/` and `w17-paper/figures/` still hold the old `fig_ladder` and the old `make_figures_w22.py`, because this spoke may not write there. A `bash build.sh` run in place in `paper-visuals/` would rebuild the old legend until `w21_figures/make_figures_w22.py` from this branch is copied into it.

## Choices logged
1. **Fig. 7 is cited twice**, once for each panel's result: panel (a) in Results/deletion, panel (b) in Results/retrained baselines. No panel letters are used, following the `Fig.~\ref{}` convention.
2. **Where the one appendix sentence goes:** as its own `\noindent` paragraph directly after Tab. cmazet in App. mazewm (Maze world-model rows), the table whose models it names, and before the figures whose panels were cut. The wording follows the brief: "The observation-only models of Tab. cmazet were not persisted, so their per-model panels are not drawn."
3. **The two consequential cuts in part 4** (the main-text "rows still pending … placeholders" sentence, and the Placeholder-index sentences) are deletions only. The ONE-sentence rule governs what part 4 adds; nothing else was added. Both statements would have been false with 0 placeholders.
4. **"about thirty … at seed 0"** in item 2 repeats the rounded phrase already accepted for Sec. 4.3 in item 5 (pw2-draft-numbers ROUNDED row 23), so the list and Results say the same thing. "seed 0" is an identifier.
5. **K=40 got a caption sentence.** No float carried K=40 / 0.991 / 0.0648, so, as in item 5, one sentence was added to the nearest caption, the validation sweep Tab. kcurve. The prose keeps the w12r qualifier "the one configuration at $K\ge36$ whose composition was measured".
6. **"more than a tuple"** replaces 1.47 (record +1.47, rule 1.47 > 1). I preferred a floor statement to "nearly one and a half".
7. **The `% PW2-4` pointer comments write "TODO" and "placeholder" without the macro backslash**, so no comment is counted as a macro use. The old comment "% The two PLACEHOLDERs name …" was reworded to say they were cut.
8. **No merge of `pw1/visuals-tooling`.** The figure pipeline in this tree (`w21_figures/`) is byte-identical to `paper-visuals/`: `make_figures_w22.py`, `make_figures.py`, `make_figures_w21.py`, `build.sh`, `figroot.py` and `cmazet_panels.py` were compared, and `data/` is equal. The build was run from a copy of `paper-visuals/` in the folder layout, which also carries the vendored engine.

## What did NOT run
- **No write** to `~/clu/outputs/paper-visuals/`, `w17-paper/`, `outputs/pw1-*`, `outputs/w*/`, TABLES.md, the registry or the claims matrix. Those folders keep the old ladder legend (Head note 4). Only `fig_ladder.pdf` was installed into the live folders; the PNG was not, since the live `figures/` hold PDFs only.
- **No float was moved to the appendix.** All four figures had a result sentence. Nothing was trimmed or shortened for length.
- **No page-by-page visual review** of the full PDFs. Checks are:
  - source-level: `CHECKS.md`, style diff, provenance, Head-held, labels, NUMBERS-LEFT, ROUNDED;
  - the build logs;
  - a rendered spot check of every edited sentence (`RENDERED.txt`, pypdf);
  - the quote font-size measurement;
  - a visual check of the new `fig_ladder.png`.
- **No figure/table index or per-section page counts regenerated.** That is the Hub's final step (f). Pages: v2 42, style 43.
- **No edits outside the brief:** not to Head-held text, appendix headings, other captions, or the preamble convention comments.
- **No compute, fits or GPU.** The figure rebuild reads recorded artefacts only.

## Deliverables in this directory
- Checks:
  - `CHECKS.md`: `\TODO` / `\PLACEHOLDER` named before → after, absolute paths, Figs. 5–8 citations, quote, labels; 0 FAIL;
  - `STYLE-DIFF-CHECK.md`;
  - `PROVENANCE-COMPARE.txt`, `prov-{before,after}-*.txt`;
  - `COMPILE.md`;
  - `RENDERED.txt`, `QUOTE-SIZE.txt`.
- Edit and number records:
  - `EDITS.md`: all 20 edits, exact before/after;
  - `LIMITATIONS.md`, `NUMBERS-LEFT-DISCUSSION.md`, `NUMBERS-LEFT.md`, `NUMBERS-LEFT-before.md`;
  - `ROUNDED.md`, `controls/ROUNDED-planted.md`.
- Hashes and Head-held: `INSTALLED-SHA256.txt`, `HEAD-HELD.md` (unchanged copy).
- Figure: `fig-*/` (builds `base-d{1,2}`, `edit-d{1,2}`, their build logs and `LADDER-TEXT-DIFF.txt`).
- Working directories: `snap-start/`, `base-*/`, `stage-*/`, `verify-live-*/`, `work/`, `LOG.md`.

## Branch and bundle
Tooling commit **5594b5e** on `pw2/draft-leftovers`: `pw2_draft_leftovers/` and the one-line legend edit in `w21_figures/make_figures_w22.py`. Outputs are not committed. Bundle `pw2-draft-leftovers.bundle`, sha256 8831ec4b…b875. `git bundle verify` exits 0 with complete history, and the head is refs/heads/pw2/draft-leftovers = 5594b5e. At the end, the bib is b2cf715f… in both folders and the live files still match `INSTALLED-SHA256.txt`.
