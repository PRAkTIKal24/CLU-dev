# FINDINGS — pw1-draft-template (ICLR 2027 template, submission mode, zero content change)

Session `pw1-draft-template` · branch `pw1/draft-template` (from `pw1/draft-final` 513e5ec) · batmac · start 2026-09-25T12:57Z.
Zero runs, zero fits, zero GPU. Two mechanical edits were installed byte-identically into both drafts. **No rendered word changed** (proof in §4).

## Status

| step | status | one line |
|---|---|---|
| 1. Latest template | **done: identical, nothing installed** | re-downloaded zip == the verified 2026-09-22 zip; all 7 members == `w17-paper/iclr2027/` == VENUE.md §3. No `iclr2027-retired-pw1/` was created (nothing to retire) |
| 2. Preamble conformity | **done** | the preamble conforms line for line with the sample. **Defect found and fixed:** the running header printed on 0 of 42 pages (the system fancyhdr v5.2 loaded instead of the template's bundled copy). **Overrides removed:** enumitem list keys, 6 lists (DRAFT-v2) / 8 (style). Everything else kept and listed, with reasons |
| 3. Double-blind audit | **done, report only; 0 leaks to fix** | /Author /Title /Subject /Keywords empty; /Creator "LaTeX with hyperref"; no XMP; the 13 embedded figures carry only relative `./figures/` names and Matplotlib Creator/Producer; 0 identity tokens in 713/714 decompressed streams; rendered text: no names, institutions, e-mails, identifying URLs, machine paths or self-reference; no acknowledgements; header now present on every page |
| 4. Rebuild and re-audit | **done** | both drafts: exit 0, 0/0 undefined, 0 errors, **0 overfull**; style diff ALL EMPTY; `\PLACEHOLDER`/`\TODO` byte-identical; provenance counts identical (one token-list detail, §3.4); `FINAL-INDEX-template.md` written |
| Acceptance: word diff EMPTY | **done** | EMPTY in both drafts, on the stage builds and on the installed PDFs; positive controls both ways |

## 1. Template hash table (downloaded vs on disk vs VENUE.md)

Download: `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`, 2026-09-25T12:57Z, HTTP 200, 39,348 B, into `template-dl-125722/`.
Zip sha256 `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`, which equals `w17-venue-scout/sources/style.zip` (the copy verified 2026-09-22). Zip member dates: `.sty` and `.tex` 2026-07-28, the other five 2025-06-25.

| member | downloaded sha256 | on disk `w17-paper/iclr2027/` | VENUE.md §3 prefix | verdict |
|---|---|---|---|---|
| iclr2027_conference.sty | 797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6 | equal | 797deef41724e937 | identical |
| iclr2027_conference.bst | 2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5 | equal | 2d67552db7ed38cc | identical |
| iclr2027_conference.tex | 03e556d6e5593e498fd39f262ec2d184ebfe8693cbf47fb7cea3402d8d5166ac | equal | 03e556d6e5593e49 | identical |
| iclr2027_conference.bib | cdd86e7d4c31854dcf2145871657c944588a6d44c3b72e160ff4baa8df1a52fb | equal | cdd86e7d4c31854d | identical |
| math_commands.tex | 90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9 | equal | 90473c4d0542070d | identical |
| natbib.sty | 88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d | equal | 88bc70c0e4846193 | identical |
| fancyhdr.sty | b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea | equal | b56ec4434b9f4607 | identical |

**Result: identical. Nothing was installed or moved.** The on-disk `iclr2027/` is unchanged (hashes in `SHA256-final.txt`).

## 2. Preamble against the sample `iclr2027_conference.tex`

| sample line | DRAFT-v2 (both drafts identical in preamble code) | verdict |
|---|---|---|
| `\documentclass{article}` | same | conform |
| `\usepackage{iclr2027_conference,times}` | same, now preceded on the same line by `\RequirePackage{iclr2027/fancyhdr,iclr2027/natbib}` (§2.1) | conform (the template's own files now load, as in the sample build) |
| `\input{math_commands.tex}` | same (found through `\input@path{{iclr2027/}{./}}`) | conform |
| `\usepackage{hyperref}`, `\usepackage{url}` | same, no options, no `\hypersetup` | conform |
| `\author{...}` | `\author{Anonymous authors}`; rendered "Anonymous authors / Paper under double-blind review" | conform |
| `%\iclrfinalcopy` | commented out | conform |
| `\bibliography{..}` + `\bibliographystyle{iclr2027_conference}` | `\bibliography{references}` + `\bibliographystyle{iclr2027/iclr2027_conference}`; the `.blg` reads "The style file: iclr2027/iclr2027_conference.bst" | conform (resolves to the template's `.bst`) |

### 2.1 Defect: the running header did not print (fixed)
- **Before:** "Under review as a conference paper at ICLR 2027" was on **0 of 42 pages** in both drafts: an empty header rule (image `var2-*/png/V0-DRAFT-v2-style-p3.png`). The template sample built from its own folder shows it on 7 of 7 pages (`sample-build-125909/`).
- **Cause (from the logs):** the drafts load the template through `\input@path`, which LaTeX consults only after kpathsea. kpathsea finds the SYSTEM `fancyhdr.sty` (2025/02/07 v5.2) and `natbib.sty` (2010/09/13 8.31b) first. The sample loads the bundled `fancyhdr.sty` and `natbib.sty` (2009/07/16 8.31). The style sets `\lhead{Under review ...}` inside `\@maketitle`'s group; the bundled fancyhdr defines `\lhead` globally, but v5.2 does not, so the header is lost when the group closes.
- **Fix (both drafts, one line, no line added):** `\usepackage{iclr2027_conference,times}` → `\RequirePackage{iclr2027/fancyhdr,iclr2027/natbib}\usepackage{iclr2027_conference,times} % fancyhdr and natbib: the template's own copies (its \lhead is global, so the running header prints)` (DRAFT-v2 l.41, style l.47).
- **After:** the header prints on 42/42 (DRAFT-v2) and 43/43 (style) pages. The header-only build (Vh) changed nothing else: every `\newlabel` equal, `.bbl` equal, page count equal, and the text diff is exactly the 42 header lines. One benign LaTeX warning remains: "You have requested package `iclr2027/natbib', but the package provides `natbib'".
- CHOICE (logged): load the template's own copies, closest to the official template. Rejected: `\lhead` in the preamble (it would override the style and break `\iclrfinalcopy`), and `TEXINPUTS` (outside the `.tex`; the w17 build recipe would silently lose it).

### 2.2 Style-parameter overrides: removed

The style file sets `\leftmargin 3pc`, `\itemsep 2pt plus 1pt minus 0.5pt`, `\topsep`, `\parsep` and `\@listi`–`\@listvi`. The enumitem keys `leftmargin=*`, `itemsep=…` and `nosep` override them, so they are removed. The page effect of each removal, measured alone against V0 (`var2-130637/EFFECT.md`):

| draft | line | removed | total pages | AI-use statement page (end of main text) | labels whose page moved |
|---|---|---|---|---|---|
| DRAFT-v2 | 230 | `\begin{enumerate}[leftmargin=*,itemsep=2pt]` (contributions, Sec. 1.3) | 42 → 42 | 18 → 18 | 0 |
| DRAFT-v2 | 1101 | `\begin{enumerate}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 18 → 18 | 0 |
| DRAFT-v2 | 2034 | `\begin{itemize}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 18 → 18 | 0 |
| DRAFT-v2 | 2075 | `\begin{enumerate}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 18 → 18 | 0 |
| DRAFT-v2 | 2414 | `\begin{enumerate}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 18 → 18 | 1: `app:placeholders` 40 → 41 |
| DRAFT-v2 | 2455 | `\begin{itemize}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 18 → 18 | 0 |
| style | 214 | `\begin{itemize}[leftmargin=*,nosep]` (three properties, Sec. 1.2) | 42 → 42 | 19 → 19 | 0 |
| style | 230 | `\begin{enumerate}[leftmargin=*,itemsep=2pt]` | 42 → 42 | 19 → 19 | 2: `sec:hole` 13 → 14, `sec:clue` 17 → 18 |
| style | 1097 | `\begin{enumerate}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 19 → 19 | 1: `sec:clue` 17 → 18 |
| style | 1972 | `\begin{itemize}[leftmargin=*,nosep]` | 42 → 42 | 19 → 19 | 0 |
| style | 2025 | `\begin{itemize}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 19 → 19 | 0 |
| style | 2066 | `\begin{enumerate}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 19 → 19 | 0 |
| style | 2405 | `\begin{enumerate}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 19 → 19 | 0 |
| style | 2446 | `\begin{itemize}[leftmargin=*,itemsep=1pt]` | 42 → 42 | 19 → 19 | 0 |
| **DRAFT-v2, all six + header (installed)** | | | **42 → 42** | 18 → 18 | 1: `app:placeholders` 40 → 41 |
| **style, all eight + header (installed)** | | | **42 → 43** | 19 → 19 | 3: `sec:hole` 13 → 14, `sec:clue` 17 → 18, `sec:followup` 18 → 19 |

Main text (marker method C, title to "AI use statement"): DRAFT-v2 **17.92 → 17.95 pp**, style **18.09 → 18.25 pp**, against the 9-page limit. The full before → after page table is in `FINAL-INDEX-template.md`.

### 2.3 Commands kept, with reasons (none overrides a parameter the style file sets)

| command | where | why kept |
|---|---|---|
| `\makeatletter\def\input@path{{iclr2027/}{./}}\makeatother` | preamble l.38–40 | a search path, not a layout parameter; it is how the w17 layout reaches the template (VENUE.md §3: "put the draft in the same folder … or add that folder to TEXINPUTS") |
| `\usepackage{amsmath,amssymb,booktabs,multirow,array,xcolor,graphicx,tabularx,longtable,adjustbox}` | preamble | standard packages; none sets a page, font or spacing parameter |
| `\usepackage{enumitem}` (now without keys) | preamble | **tested:** a build with it removed gives page content streams equal on 42/42 (DRAFT-v2) and 43/43 (style) pages and all labels equal |
| `\setlength{\tabcolsep}{2–3pt}` ×19 | each inside a `table` float (checked: the enclosing environment of all 19) | local to the float; `\tabcolsep` is not a style-file parameter |
| `\small` / `\scriptsize` after `\centering` in tables; `\small` in the `\figslot` "figure pending" box | floats | local float type size; the style defines these sizes and sets no float-size rule |
| `adjustbox{max width=\linewidth}` around tabulars | floats | scales a wide table to the text width; it changes no parameter |
| `\medskip` | between two tabulars inside one table float (l.1358) | a positive skip inside a float |
| `\clearpage` before `\appendix` | l.1253 | a page break, not a parameter |
| `\begin{quote}\small` — the deletion-scope statement | Sec. 2, DRAFT-v2 l.334 / style l.336 | **CANNOT TELL, listed and left in**: a local 9 pt size in body text. It is not a style-file parameter, but the template's body type is 10 pt. For the Head |
| float placement `[t]`, `[h]` (LaTeX changes `h` to `ht`), `[p]` | floats | placement options, not parameters |

Searched and absent (0 in either draft): `geometry`, `\textwidth`/`\textheight`/margin settings, `\fontsize`, `\linespread`/`\baselinestretch`, `\parskip`/`\parindent`, `caption`/`titlesec`/`\captionsetup`/`\titlespacing`, `\abovecaptionskip`/`\belowcaptionskip`, float parameters (`\topfraction`, `\textfraction`, `\floatsep`, `\textfloatsep`, `\intextsep`, `\dbl…`), `\vspace` (any sign), `\vskip`, `\enlargethispage`, `\looseness`, `\sloppy`/`\emergencystretch`/penalties, `\arraystretch`, `\pagestyle`/`\thispagestyle`, negative `\hspace`, negative `\\[…]`.

## 3. Checks (stage `var2-130637/checks-Vhl/CHECKS.txt`; installed tree `final-130917/`)

### 3.1 Compile
| build | draft | exits | passes | undefined cit. / ref. | errors | overfull | underfull | bib items | pages |
|---|---|---|---|---|---|---|---|---|---|
| scratch (clean w17 tree) | DRAFT-v2 | 0, 0, 0 0 0 | 4 | 0 / 0 | 0 | **0** | 14 (was 15) | 55 | 42 |
| scratch | style | 0, 0, 0 0 0 | 4 | 0 / 0 | 0 | **0** | 13 (was 12) | 55 | 43 (was 42) |
| in place `w17-paper/` | DRAFT-v2 | 0, 0, 0 | 2 | 0 / 0 | 0 | 0 | 14 | 55 | 42 |
| in place | style | 0, 0, 0 | 2 | 0 / 0 | 0 | 0 | 13 | 55 | 43 |

`.aux`, `.bbl`, `.out` and PDF text: in place == scratch in both drafts. Installed PDF text == stage (Vhl) text in both drafts. Overfull boxes: none to list.

### 3.2 Style diff-check (`pw1_draft_sweeps/style_diff_check.py`, DRAFT-v2 vs style): **ALL DIFFS EMPTY** (numbers, placeholders, TODOs, artefact paths, provenance order, headings, labels, floats, preamble).

### 3.3 `\PLACEHOLDER` / `\TODO`: arguments byte-identical and in the same order, before and after, in both drafts (2 PLACEHOLDER, 4 TODO each). Lines unchanged (no line was added).

### 3.4 Provenance (`check_provenance.py` + `check_derived.py`)
| draft | checked number tokens | unsourced lines | not-found tokens | derived-verified | before → after |
|---|---|---|---|---|---|
| DRAFT-v2 | 2597 | 7 | 9 | 1 (106 = 317 − 211) | all counts identical |
| style | 2597 | 7 | 10 | 1 | all counts identical |

**Not byte-identical, printed:** the unsourced-line finding for l.41 (style l.47) reads `['027']` before and `['027', '027', '027']` after. The checker reads the digits of the directory name `iclr2027/` in the two new package paths as number tokens, on a line it already flagged for the same token in `iclr2027_conference`. No result number was added. The fix went on the existing line so that no new unsourced line (7 → 9 in a first two-line version, rejected) and no line-number shift appears. The derived check output is byte-identical.

### 3.5 Labels: 75 → 75 per draft; 0 renumbered; 0 float numbers changed; 0 float pages moved. Section-label pages moved: DRAFT-v2 `app:placeholders` 40 → 41; style `sec:hole` 13 → 14, `sec:clue` 17 → 18, `sec:followup` 18 → 19.

### 3.6 Residual audit (pw1-draft-final's 55 rules, src and pdf): per-rule hits **identical** to pw1-draft-final's (the JSONs are equal except the `ctx` snippets, which now contain the header line or moved floats). Figure folders: 0 problems; both READMEs == regenerated.

## 4. Acceptance: the rendered word diff (the "no content change" proof)

Tool: `pw1_draft_template/worddiff.py BEFORE.pdf.txt AFTER.pdf.txt` (text by `pw1_draft_cuts/pdftext.py`, pypdf). It removes page furniture per page: the line-number ruler (the page's own range only), the running header line, and the page number. It joins line-end hyphens and runs a word-level difflib. Opcodes whose two sides are equal once hyphens and spaces are removed are HYPHENATION/EXTRACTION. The rest go to a character-level move resolution: repeatedly remove the longest common substring of at least 20 characters; the leftovers on each side, joined in order, must be identical. The order-free character multiset must also be equal.

| comparison | draft | word opcodes | hyphenation/extraction | moved blocks (chars) | leftover before = after | char multiset | RESULT |
|---|---|---|---|---|---|---|---|
| installed before → installed after (`WORDDIFF-installed-*.txt`) | DRAFT-v2 | 20 | 16 | 2 (454) | `['•']` = `['•']` | equal (143,103 chars) | **EMPTY** |
| installed before → installed after | style | 47 | 15 | 16 (4,539) | `['5–6).', 'Table', 'dition']` = same | equal (144,042 chars) | **EMPTY** |
| stage V0 → Vhl (`checks-Vhl/WORDDIFF-*.txt`) | both | same as above | | | | | **EMPTY** |

The moved blocks are text that crosses a float differently: a bullet of "Declared not-runs" ("LACE and PoE refit on …", l.2463) that now sits after Tab. 20 in the text stream in DRAFT-v2; in the style draft, paragraphs of the appendices "Maze world-model rows", "The encoder: adaptation and the unfreezing decomposition" and "The two laws on this study's n-CLU, stated in full" that now fall on the other side of a table float (the appendices run one page longer). The style draft's leftovers are a word split by a float ("con-" | Table 20 | "dition").
Positive controls: `--plant` (one word replaced and one deleted in AFTER) → **NOT EMPTY**, leftover `'the120,028•'` vs `'PLANTEDWORD•'` (DRAFT-v2); the style draft likewise. `--plant-move` (a 60-word block moved 3,000 words on) → **EMPTY**, 1 moved block.
Furniture removed after: the header 42 (DRAFT-v2) / 43 (style) lines; before: 0.

## 5. Double-blind audit (report; nothing in the prose edited)

| check | DRAFT-v2 | style | action |
|---|---|---|---|
| PDF /Author, /Title, /Subject, /Keywords | all empty | all empty | none needed |
| /Creator, /Producer | "LaTeX with hyperref", "pdfTeX-1.40.29" | same | none |
| /PTEX.Fullbanner | "… (TeX Live 2026/Homebrew) kpathsea version 6.4.2" | same | not identifying; printed |
| /CreationDate, /ModDate | carry the local offset `+02'00'` | same | a time zone, not an identity; printed for PW-2's anonymised upload |
| XMP metadata | none | none | — |
| embedded figures (PTEX.FileName) | 13, all relative `./figures/…` | 13 | not identifying |
| embedded figures' PTEX.InfoDict | Creator "Matplotlib v3.10.8, https://matplotlib.org", Producer "Matplotlib pdf backend v3.10.8" (plus empty dicts) | same | not identifying |
| identity tokens in all 713 / 714 decompressed streams and strings (`batmac`, `/Users/`, `forgis`, `clu/`, `chlu`, `CHLU`, `home/`) | 0 | 0 | — |
| `github.com/` | 3 hits: the 3D Shapes dataset reference (google-deepmind/3d-shapes), text and link | same | a cited third-party dataset |
| rendered names / institutions / e-mails | none outside the bibliography (cited authors only) | none | — |
| rendered URLs | only in the references (DeepMind dataset, OpenReview) | same | — |
| rendered machine paths, file names, internal run names | 0 | 0 | — |
| "our previous / prior / earlier work", "we previously" | 0 | 0 | — |
| acknowledgements | none (0 hits) | none | — |
| author block | "Anonymous authors / Paper under double-blind review" | same | — |
| running header | **was missing; now "Under review as a conference paper at ICLR 2027" on every page** | same | fixed at formatting level (§2.1) |
| "anonymised" in prose | 2: the reproducibility statement ("An anonymised code archive will accompany the submission…") + its `\TODO` | same | prose, Head's; not edited |

**Venue statements (MF-12, HELD, unchanged):** the `\subsubsection*` headings "AI use statement" (DRAFT-v2 l.1222, p. 18; style l.1211, p. 19; it holds a `\TODO`), "Ethics statement" (l.1228 / 1217) and "Reproducibility statement" (l.1236 / 1225). They are in that order, after Sec. 6 and before the references.
**Machine paths in `.tex` comments (not stripped; PW-2's anonymised-upload step):** per draft, 618 / 624 comment-bearing lines. 453 carry a path (601 / 602 path tokens), of which 442 / 443 are `% SRC:` lines with relative artefact paths. **2 carry a home path:** DRAFT-v2 l.12 `~/clu/evidence/outputs/` and l.35 `~/clu/outputs/w17-paper/` (style l.18, l.41). None contains a user name.

## 6. Hashes
- Before (live, = pw1-draft-final's installs): DRAFT-v2.tex `ef88256e…`, style `6f6925f8…`; PDFs `a5067a19…`, `d7f2512e…` (`before/SHA256-before.txt`).
- Installed `.tex` (== stage Vhl, cmp-verified): DRAFT-v2 `f9642d0fb44e526c0ea0fbe104395670054d301b6c0a651ce14969633d5e6048`, style `27babec26b977d96cbc481027f275485a9eeeec891fe9dbf6fb3423b8488054b`.
- Installed PDFs (in-place build; text == stage text; bytes differ from the stage only by date/ID): DRAFT-v2 `c05f96897a3a0907b48eaae1ffbbae129f718ab8a3711f86260c77d071fcb42b`, style `b03544518b8320c8a0aee16ff884ea607e1d767e78ed12d2860f9b42cd560a2e` (`SHA256-final.txt`, with the unchanged `iclr2027/` members).

## For the Head (printed, not changed)
1. `\begin{quote}\small` (the deletion-scope statement, Sec. 2) sets body text at 9 pt. It is not a style-file parameter, so it was left in; removing `\small` would be layout-only.
2. The style draft is now 43 pages (the template's list spacing), and its main text is 18.25 pp. DRAFT-v2 stays at 42 pp, with 17.95 pp of main text. MF-11 is held.
3. The PDF dates carry `+02'00'`. PW-2's anonymised upload may want `SOURCE_DATE_EPOCH` / `\pdfinfoomitdate`, which are not applied here.
4. The 2 home paths in the `.tex` comments (§5) are for PW-2's strip.

## What did NOT run
- No content edit: no sentence, number, table cell, caption, figure, citation or section order changed. The source diff is 7 lines (DRAFT-v2) and 9 lines (style), all shown in §2.1–2.2.
- No template install and no `iclr2027-retired-pw1/`, because the download was identical.
- No prose edit for double-blind reasons (none was needed at formatting level); the `.tex` comments were not stripped; the venue statements were not touched (MF-12).
- `\small` in the quote was not removed (undecidable; listed).
- No figure was rebuilt. The figure folders were only audited (0 problems).
- The pdf channel has pypdf's known blind spots (spacing glitches, `\_` as a space). The word diff treats spacing-only differences as extraction artefacts, and the character-stream checks cover them.
- `pdfinfo` and `pdftotext` are not installed on batmac, so all PDF reading used pypdf / pypdfium2 (images in `var2-*/png/`).
- No measurement, fit, model load or GPU use. `vm_stat` showed ≥176k free pages (~2.9 GB) before the builds.
- `outputs/pw1-advisor-report.md` was not written: it is the Hub's.
- The reading copy was not mirrored to another machine (no path named). The batmac path is `~/clu/outputs/w17-paper/`.

## Branch and bundle
Branch `pw1/draft-template` at `a4de290` (tooling only: `pw1_draft_template/`, 8 files; no outputs committed). Bundle `pw1-draft-template.bundle` sha256 `f4e7371054ea3d521dc1ba6d3dcde92d1bb49052286bbd42a2fb0c0763121dbb`; `git bundle verify`: okay, complete history.
