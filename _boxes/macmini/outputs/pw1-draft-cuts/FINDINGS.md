# FINDINGS — pw1-draft-cuts (PW-1 plan items 14, 15, 17)

Session `pw1-draft-cuts` · branch `pw1/draft-cuts` (from `pw1/draft-sweeps` d80809f) · batmac · start 2026-09-25T11:20:13Z.
Tooling (committed, `pw1_draft_cuts/`): `cuts.py` (the cut, 38 anchored operations per draft, each asserting a unique anchor),
`scan_cuts.py` (residual scan of the item tokens), `report.py` (removal table, refs, placeholders, index, pages), `pdftext.py`.
Reused from `pw1_draft_sweeps/`: `compile_check.sh`, `style_diff_check.py`, `check_provenance.py`, `scan.py`.
Snapshot: `snap-before/` · stage: `stage-112956/` · builds: `build-before-112332/` (before), `build-final-112957/` (installed).

## Status per assigned item

| item | status | note |
|---|---|---|
| 14 — the maze section is CMAZE-T only | **done** in both drafts | CMAZE-2 deletion transfer + `tab:maze-del`, CMAZE-2b request layer, route/read verb + `tab:verb` + `fig_maze_readverb` (`fig:verb`), the `fig_maze_requestlayer` note, `tab:maze-del-2b`, `tab:req`, `tab:req2b`, CMAZE-O paragraph + `tab:cmazeo` + its harness smoke / ladder / negatives text, the three appendices (CMAZE-2 in full, CMAZE-1 proper, the read verb in full), and every abstract / contribution / thickness / limitation / leaving-note sentence resting on them are out. ONE paragraph "Toward a CMAZE suite" closes the CMAZE-T section and cites no removed result. CMAZE-1/2/2b/O: 0 in rendered source and 0 in the PDF text of both drafts. |
| 15 — Fig. `maze-rel-tuples` out | **done** | slot + caption removed (inside the read-verb appendix span); its one `\ref` (conditions-figure caption) now points at Tab. `cmazet-sym`, which carries the seed-4 control tie and the seed-5 baseline tie. The text's one-sentence tie statements (relational paragraph) are unchanged; there was no other text discussing the figure. |
| 17 — Fig. `cmazet-deletion` out | **done** | slot, caption and its `\PLACEHOLDER` removed; its 3 `\ref`s rewritten (task-figure range now ends at `fig:cmazet-door-tint`; placeholder index two lists). The CMAZE-T deletion facts stay in Tab. `cmazet` (deletion rows) and Tab. `cmazet-sym` (deletion row), and in ONE text sentence (the "what is not claimed" paragraph); the duplicate clause "and deletion stays exact on all six seeds" in the symmetric-input paragraph was removed to leave one. That sentence still says "Exact deletion holds" — item 13's wording, not mine (see below). |

### Choices taken (open in the plan; logged)
1. **App. "Maze refusal displays (retired)" removed** although not named in item 14's list: it is CMAZE-2 route-request material, and item 14 says everything on the maze that is not CMAZE-T leaves. Limitation "Refusal on the maze was novelty detection…" went with it.
2. **Tab. `impl-maze` removed** (implementation 1 was the CMAZE-2 n-CLU). App. E renamed "Deletion on Shapes3D: two implementations, and the failure they agree on" (the word "Exact" was dropped from the heading I rewrote, per the vocabulary rule); implementations renumbered 2→1, 3→2 in Tab. `impl`, its caption ("row 3" → "row 2") and "Implementation 1 in full". The "failure agreement" paragraph already spoke of two implementations.
3. **Sec. 2 sentence on the frozen encoder**: its reason ("the frozen autoencoder's own training set contains evaluation-bank items, 543 / 529 / 552 on the three maze seeds") was a CMAZE-2 measurement; rewritten without a number: "declared outside the deletable object: they are frozen, so no deletion touches their weights, which may have been fitted with the deleted rows present."
4. **App. K-curve**: the comparison "exceeds the 0.938 at which a maze n-CLU was decorative" (CMAZE-2's rank/width, w13 TABLES "wave 9's maze arm") removed; the sentence keeps "the controls still collapse, so rank/width is a warning and only the control decides".
5. **Section structure**: the maze section has no subsection left; the world-model subsection heading is removed and `\label{sec:maze-wm}` sits on the section beside `sec:maze` (the Head's abstract `\TODO` references `sec:maze-wm` and stays byte-identical and resolving). Section title unchanged ("The same properties on a maze"). New two-sentence lead-in (no numbers).
6. **The CMAZE-T conditions figure (`fig:maze-wm`, item 1's)** physically sat inside the removed read-verb appendix. It was moved verbatim (with its provenance block) to the head of App. "Maze world-model rows"; only its `\ref{fig:maze-rel-tuples}` changed (→ Tab. `cmazet-sym`). It remains unreferenced from the text, as before (items 1/16 own it).
7. **Limitation 1** renamed "On the maze transition task the n-CLU's lead is small and seed-dependent" (wording of the leaving note); its CMAZE-T sentence kept verbatim. Contribution 3 keeps its title and CMAZE-T sentences; the route clause and `\ref{sec:maze}` left.
8. **App. mazewm**: "Both harnesses" → "The harness"; CMAZE-O smoke, CMAZE-O ladder, the eighteen-run count, the CMAZE-O not-run sentence, the fallback route task, the CMAZE-O negatives removed; "every across-wells cell in Tabs. cmazet and cmazeo" → "Tab. cmazet". Captions of Tabs. `cmazet` / `cmazet-sym` lose their "contains no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row" sentences (the names may not appear).
9. **Declared not-runs**: the CMAZE-2 repaired post-deletion composition item removed; "a figure for the plan-versus-latent result" dropped from the figure-slot item.
10. Figure image files untouched (`fig_maze_readverb.pdf`, `fig_maze_rel_tuples.pdf`, `fig_cmazet_deletion.pdf`, `fig_maze_requestlayer*` stay in `figures/` and `paper-visuals/`), per the brief.

### The CMAZE-suite paragraph (both drafts, verbatim)
> **Toward a CMAZE suite.** CMAZE-T is the first instrument of a planned CMAZE suite on the same engine and judge. The transition task will be extended to observation generation, to route planning and to the further tasks already prototyped, so that CMAZE as a whole becomes a useful benchmark for the world-models community.

### Abstract (Head-held `\TODO` byte-identical; the only other sentence touched)
Before (removed whole; pointer comment left naming w13-seeds-and-tables/TABLES.md and w14-read-verb/FINAL-REPORT.md):
> On a colour maze with the engine's integer judge, deletion transfers (relative residual 7.522e-16 to 8.826e-16, projector residual 0.000), while composition on routes does not transfer through the n-CLU: moving the composition verb from latent space to plan space lifts held-out validity by +0.4422 to +0.5354, but it lifts a matched-bytes lookup floor more, so the composition operator belongs to the interface.

After: (nothing; the next line is the Head's `\TODO`, unchanged). No other abstract sentence rests on removed material (the Shapes3D sentences and the zero-dynamics closing sentence are untouched). The style twin's sentence ("whereas composition on routes…") was removed identically.

Other before → after sentences (DRAFT-v2; the style twin's are in `stage-112956/CUTS-style.json`):
- Contribution 3: "Deletion transfers exactly; composition on routes is recovered by a plan-space verb, which we report as an interface result because it lifts a matched-bytes lookup floor more than it lifts the n-CLU (Sec. 4). On a transition instrument with symmetric inputs, …" → "On a colour-maze transition instrument (CMAZE-T) with symmetric inputs, …"
- Thickness: "On routes, where the engine's judge scores whole plans at three scored seeds, the result is a negative for the n-CLU and a positive for the interface. The transition instrument on the same engine, also under the engine's judge, prints six seeds …" → "The maze transition instrument, under the engine's judge, prints six seeds …"
- Leaving note: "It does not provide composition on an engine-judged route task, where the operator is task-shaped and belongs to the interface; on the transition task its lead …" → "On the maze transition task its lead …"
- Limitation 1: "No n-CLU row on the route tasks (CMAZE-2, CMAZE-2b). On CMAZE-2 a matched-parameter transformer beats the n-CLU on every composition row (−0.0560 / −0.0209 / −0.0434), … (App. J). On CMAZE-2b the controls again fail to collapse (Sec. 4.3). On CMAZE-T …" → "On the maze transition task the n-CLU's lead is small and seed-dependent. On CMAZE-T …"

## Acceptance — builds and checks (both drafts, built from the w17-paper layout)

| check | before (`build-before-112332`) | after (`build-final-112957`, installed) |
|---|---|---|
| exit codes (pdflatex, bibtex, pdflatex ×3) | 0 | 0 |
| undefined citations / references | 0 / 0 | 0 / 0 |
| LaTeX errors / overfull boxes | 0 / 0 | 0 / 0 |
| bibliography items | 55 | 55 |
| pages DRAFT-v2 / DRAFT-v2-style | 50 / 50 | **41 / 42** |
| style diff-check (numbers, placeholders, TODOs, artefact paths, provenance comments and order, headings, labels, floats, preamble) | ALL EMPTY (`STYLE-DIFF-CHECK-before.md`) | **ALL EMPTY** (`stage-112956/STYLE-DIFF-CHECK.md`) |
| provenance checker (unsourced lines / not-found tokens) | v2 7 / 8; style 7 / 9 | **identical lists** (same tokens; line numbers shifted); checked tokens 3,233 → 2,598 per draft |
| `\PLACEHOLDER` | 3 | **2** — removed: the third, "w16 CMAZE-T post-deletion per-arm generated s_{t+1} (n-CLU before / after downdate, each baseline as trained / retrained without): not buildable from persisted records; …" (it lived in the `fig:cmazet-deletion` caption, item 17). The two survivors are byte-identical and in the same order. |
| `\TODO` | 4 | 4, all byte-identical (the abstract CMAZE-T `\TODO` still references `sec:maze-wm` and `tab:cmazet-sym`, both resolve) |
| vocabulary scan (`pw1_draft_sweeps/scan.py`) | — | no token count rose in either draft (`RESID-vocab-*.txt`); config 26→18, exact 114→94 fell with the removed text |

Every number that left the text keeps a pointer: each removed or rewritten block with numbers carries a `% pw1-draft-cuts (item N): removed …; its numbers stay in: <records>` comment listing the block's own provenance paths (plus w13 TABLES for the 0.938 rank/width and the CMAZE-2 post-deletion not-run, and w16 FINDINGS/TABLES for the CMAZE-O ladder numbers, each checked present in those files: 0.938, 0.0208, 1,232,064, 0.1367, 0.3967, 1.02, 0.99, 30,000, 0.463, 0.1024, 543). The pointer comments avoid the provenance keyword, so the style diff-check compares only surviving provenance comments. Numbers that left without a pointer are only renumbering / names (implementation "2."/"3.", "CMAZE-1," in captions, "colour 1"/"seed 0" of the removed deletion-figure caption, which names its records in its own comment).

## Acceptance — residual scan

Source (rendered code, comments excluded; `\ref`/`\label` keys INCLUDED so a dangling key would count), `scan_cuts.py`, both drafts identical counts:

| token | before | after | reason for any residual |
|---|---|---|---|
| CMAZE-1 | 14 | 0 | |
| CMAZE-2 | 24 | 0 | |
| CMAZE-2b | 26 | 0 | |
| CMAZE-O | 18 | 0 | |
| read verb | 3 | 0 | |
| verb | 28 | 0 | |
| route | 44 | 7 | 6 × the English "route" in the Shapes3D deletion sense ("only exact route is the retrain", "Our route is exact", Sec. 3.5's "deletion route" ×4 lines); 1 × "route planning" in the mandated CMAZE-suite paragraph |
| `MERGE`/`AVG` | 22 | 0 | |
| request layer | 6 | 0 | |
| maze-rel-tuples (key or file) | 3 | 0 | |
| cmazet-deletion (key or file) | 5 | 0 | |
| readverb | 1 | 0 | |
| requestlayer | 0 | 0 | (only ever in a comment; that comment left with the read-verb appendix) |
| bare "CMAZE" | 0 | 3 | the suite paragraph (heading, "CMAZE suite", "CMAZE as a whole") |

PDF text (pypdf over the installed PDFs), both drafts: CMAZE-1 14→0, CMAZE-2 (incl. 2b) 50→0, CMAZE-2b 24→0, CMAZE-O 18→0, read verb 2→0, route 43→7 (the same 7 as above), maze-rel-tuples / cmazet-deletion / readverb / requestlayer 0→0. "verb" 18→1: "verbatim" (the inference-note heading). No remaining figure image carries a CMAZE-1/2/2b/O string.

References: `\ref` tokens 187 → 152 (v2), 188 → 152 (style); keys referenced but not labelled after: **none** in either draft; 0 undefined references in both builds. Full per-key list: `stage-112956/CUTS-REPORT.md` §2.

## Installed

Lock guard before install: clean (live: pw1-plot-shapes, pw1-render-shapes; no ABORT pattern). Live drafts and PDFs were cmp-equal to `snap-before/`. Installed byte-identically (cmp OK at destination; `installed/SHA256-install.txt`):

| file | before (snap-before/SHA256SUMS) | after |
|---|---|---|
| DRAFT-v2.tex | `a277235db5ad2d434fbca1efe438a8620dde6461fc0bcb6b38d4a2512ac3fb12` | `9a5dec5ce0aad756aea9fe674dc8dc387a7f3d533d87eb858f6065b0b6634c33` |
| DRAFT-v2-style.tex | `53063ef5818cc26f2b53dccd083b678435ab81d1a3f30a5b06911ee21f4e8095` | `d635aead03f7f7d0b9a487d70a839466aa03a43f2e537447c2db3a7b9a953e89` |
| DRAFT-v2.pdf | `58004b79442c0c2365478ac1b9af69767ccf36b00f34ce348942a8040e2d548e` | `fba66b9ba4dfe16127620ce0fb84c56b36e77d35e3e610b6ca10337e79740ae1` |
| DRAFT-v2-style.pdf | `dd4092108468272e97ad63732ecb301ddc43668df5f01a7c97abf6e28b588f71` | `a03ada380372509fb7a7319420d2c42d8057d1c3fa8b13b8d23987e946fb99d5` |

Only the two `.tex` and the two PDFs were installed; the stale `.aux/.bbl/.log/.out` in `w17-paper/` were not replaced.

## For the Head / next spokes
- Item 13 (not mine) still owns "exact" in the deletion sense; the surviving CMAZE-T deletion sentence reads "Exact deletion holds on words plus observation on every seed: …", and Tab. `impl` / Tab. `cmazet` / Tab. `cmazet-sym` still say "Exact (n-CLU-scoped)" / "n-CLU-scoped exactness".
- Items 1 and 8 (not mine): the conditions-figure caption still carries WIN/COMPARABLE/LOSS rule text, the ceiling and "Seeds 1 and 4 are hard"; the text keeps its ceiling paragraph. Untouched.
- `fig:maze-wm` and the four CMAZE-T panel figures are not referenced from the body text (unchanged from before); the figure spokes may want a `Fig.~\ref` from Sec. 4.
- The maze section is now ~1.3 pages (p. 13–14 in DRAFT-v2); the last main-text section (Follow-up) now starts on p. 16 (was p. 19).
- Retiring the image files `fig_maze_readverb`, `fig_maze_rel_tuples`, `fig_cmazet_deletion`, `fig_maze_requestlayer*` from `figures/` and `paper-visuals/` (and their README entries) is left for the figure spoke, per the brief.

## What did NOT run
- No figure image, figure script or `paper-visuals/` file was touched; the removed figures' image files remain on disk.
- No item outside 14, 15, 17 was applied: no "exact" sweep (item 13), no ceiling / hard-easy / WIN-label edits (items 1, 8), no panel-text edits (item 16), no wording from item 7.
- No bibliography change; bib items stayed 55 (no citation lived only in the removed text).
- The registry, the claims matrix, TABLES.md and every `outputs/w*/` directory were only read.
- No measurement, fit or GPU run of any kind.
- Section-level page spans in `CUTS-REPORT.md` §5 are from the .aux start pages (a section's span is up to the next section's start page), not a measured per-section length.
- The stale `.aux/.bbl/.log` in `w17-paper/` were not rebuilt in place.
- pdftotext is not installed on batmac; the PDF text scan used `uv run --with pypdf`.

---

# Appendix: the generated evidence (`stage-112956/CUTS-REPORT.md`, verbatim)

## 1. Removal table (every block removed or rewritten, per draft)

Line ranges are in the snapshot drafts (`snap-before/`). Words = rendered-word estimate (comments, control words and braces dropped). `kind`: remove (text gone, pointer comment left), replace (rewritten), move (relocated, not removed).

| # | item | kind | what | v2 lines | v2 words −/+ | style lines | style words −/+ | refs removed |
|---|---|---|---|---|---|---|---|---|
| 1 | 14 | remove | abstract sentence: deletion transfers on a colour maze; route composition and the plan-space verb | 125–130 | 64 / 0 | 128–132 | 64 / 0 | — |
| 2 | 14 | replace | contribution 3: the maze deletion-transfer and route/verb clause | 252–255 | 46 / 15 | 251–254 | 46 / 15 | `sec:maze` |
| 3 | 14 | replace | thickness paragraph: the route-task sentence | 270–272 | 42 / 11 | 269–271 | 42 / 11 | — |
| 4 | 14 | replace | object section: the CMAZE-2 autoencoder evaluation-bank overlap (543 / 529 / 552) | 350–353 | 47 / 30 | 351–353 | 45 / 30 | — |
| 5 | 14 | replace | maze section lead-in (CMAZE-1/2/2b/O names), Sec. maze-del with Tab. maze-del (CMAZE-2), Sec. req-layer (CMAZE-2b), Sec. verb with Tab. verb… | 926–1107 | 1573 / 50 | 921–1101 | 1584 / 50 | `sec:maze-wm`, `tab:maze-del`, `sec:maze-wm`, `sec:verb`, `tab:maze-del`, `tab:maze-del-2b`, `app:verb`, `tab:req`, `tab:req2b`, `app:verb`, `app:verb`, `tab:maze-del`, `sec:shapes`, `tab:cmazeo` |
| 6 | 17 | replace | CMAZE-T symmetric-input paragraph: the second deletion clause (the deletion sentence of the "what is not claimed" paragraph stays, with the … | 1132–1134 | 22 / 14 | 1126–1128 | 22 / 14 | — |
| 7 | 14 | replace | the CMAZE-O paragraph (withdrawn margin, control check) and the maze section's closing summary (CMAZE-2/2b deletion transfer, the verb, CMAZ… | 1179–1205 | 327 / 82 | 1173–1199 | 327 / 82 | — |
| 8 | 14 | replace | limitation 1: "no n-CLU row on the route tasks" (CMAZE-2, CMAZE-2b) | 1216–1221 | 67 / 17 | 1211–1216 | 67 / 17 | `app:cmaze2`, `sec:verb` |
| 9 | 14 | remove | limitations: post-deletion composition on the maze (CMAZE-2) and maze refusal as novelty detection | 1251–1261 | 124 / 0 | 1246–1256 | 124 / 0 | — |
| 10 | 14 | replace | leaving note: composition on an engine-judged route task | 1322–1323 | 33 / 14 | 1312–1313 | 33 / 14 | — |
| 11 | 14 | replace | App. K-curve: the CMAZE-2 n-CLU rank/width comparison (0.938) | 1561–1563 | 33 / 19 | 1549–1551 | 33 / 19 | — |
| 12 | 14 | replace | App. implementations: section heading (three -> two implementations) | 1666–1666 | 10 / 11 | 1654–1654 | 10 / 11 | — |
| 13 | 14 | remove | App. implementations: Tab. impl-maze (implementation 1, the CMAZE-2 n-CLU) | 1669–1684 | 53 / 0 | 1657–1672 | 53 / 0 | `tab:impl` |
| 14 | 14 | replace | App. implementations: Tab. impl caption (renumbered; the comparison with the maze band) | 1688–1691 | 39 / 27 | 1676–1679 | 39 / 27 | `tab:impl-maze` |
| 15 | 14 | replace | App. implementations: Tab. impl rows renumbered | 1699–1699 | 10 / 10 | 1687–1687 | 10 / 10 | — |
| 16 | 14 | replace | App. implementations: Tab. impl rows renumbered | 1700–1700 | 7 / 7 | 1688–1688 | 7 / 7 | — |
| 17 | 14 | replace | App. implementations: paragraph heading renumbered | 1706–1706 | 4 / 4 | 1694–1694 | 4 / 4 | — |
| 18 | 15 | replace | Fig. maze-wm caption: the \ref to Fig. maze-rel-tuples (the tie keeps its table pointer) | 2120–2120 | 12 / 12 | 2108–2108 | 12 / 12 | `fig:maze-rel-tuples` |
| 19 | 14 | move | App. read verb: the CMAZE-T conditions figure (fig:maze-wm) lifted out of the removed appendix, re-seated in App. mazewm | 2097–2123 | 400 / 0 | 2085–2111 | 400 / 0 | — |
| 20 | 14 | remove | App. CMAZE-2 in full (Tab. cmaze2), App. CMAZE-1 proper (Tab. cmaze1), App. the read verb in full (Tabs. req, req2b, maze-del-2b; Fig. verb … | 1984–2285 | 2145 / 0 | 1972–2273 | 2165 / 0 | `tab:cmaze2`, `tab:req2b`, `sec:req-layer`, `tab:req`, `tab:maze-del-2b`, `sec:maze-del`, `tab:maze-del`, `sec:hole` |
| 21 | 14 | move | App. mazewm: the conditions figure re-seated at the head of the CMAZE-T appendix | 2527–2527 | 0 / 400 | 2518–2518 | 0 / 400 | — |
| 22 | 14 | replace | App. mazewm: "both harnesses" (CMAZE-O harness) | 2529–2529 | 7 / 7 | 2520–2520 | 7 / 7 | — |
| 23 | 14 | remove | App. mazewm: the CMAZE-O harness smoke | 2541–2547 | 76 / 0 | 2532–2538 | 76 / 0 | — |
| 24 | 14 | replace | App. mazewm: control defect, CMAZE-O half | 2549–2549 | 9 / 9 | 2540–2540 | 9 / 9 | — |
| 25 | 14 | replace | App. mazewm: control defect, the \ref to Tab. cmazeo | 2554–2555 | 9 / 8 | 2545–2546 | 9 / 8 | `tab:cmazeo` |
| 26 | 14 | replace | Tab. cmazet caption: the "contains no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row" sentence | 2575–2576 | 17 / 7 | 2566–2567 | 17 / 7 | — |
| 27 | 17 | replace | panel-figure comment: three -> two placeholders | 2602–2602 | 0 / 0 | 2593–2593 | 0 / 0 | — |
| 28 | 17 | replace | Fig. cmazet-deletion (fig_cmazet_deletion): slot, caption and its PLACEHOLDER | 2650–2661 | 130 / 0 | 2641–2652 | 130 / 0 | `fig:cmazet-families`, `tab:cmazet` |
| 29 | 14 | replace | Tab. cmazet-sym caption: "no CMAZE-O, CMAZE-1, CMAZE-2 or CMAZE-2b row" | 2676–2676 | 9 / 1 | 2667–2667 | 9 / 1 | — |
| 30 | 14 | remove | Tab. cmazeo (CMAZE-O, observation generation) | 2710–2739 | 488 / 0 | 2701–2730 | 488 / 0 | `tab:cmazet` |
| 31 | 14 | replace | App. mazewm, configuration ladder and run count: the CMAZE-O ladder and the eighteen-run count | 2752–2761 | 141 / 19 | 2743–2752 | 141 / 19 | — |
| 32 | 14 | remove | App. mazewm: the fallback route task paragraph (CMAZE-2b; refs to Tabs. verb and maze-del-2b) | 2763–2766 | 53 / 0 | 2754–2757 | 53 / 0 | `tab:verb`, `tab:maze-del-2b` |
| 33 | 14 | replace | App. mazewm, declared negatives: the CMAZE-O half | 2768–2772 | 57 / 4 | 2759–2763 | 57 / 4 | — |
| 34 | 17 | replace | Fig. task-cmazet caption: figure range ending at Fig. cmazet-deletion | 301–301 | 1 / 1 | 300–300 | 1 / 1 | `fig:cmazet-deletion` |
| 35 | 17 | replace | App. placeholder index: three rows -> two (Fig. cmazet-deletion) | 2885–2887 | 31 / 31 | 2876–2878 | 31 / 31 | `fig:cmazet-deletion` |
| 36 | 17 | replace | App. placeholder index: the panel-figure list (Fig. cmazet-deletion) | 2899–2901 | 16 / 16 | 2890–2892 | 16 / 16 | `fig:cmazet-deletion` |
| 37 | 14 | remove | App. not-runs: the repaired maze post-deletion composition (CMAZE-2) | 2926–2927 | 14 / 0 | 2917–2918 | 14 / 0 | — |
| 38 | 14 | replace | App. not-runs: the plan-versus-latent (read verb) figure | 2932–2932 | 15 / 11 | 2923–2923 | 15 / 11 | — |

Totals (moves excluded): v2 5731 words removed, 437 written; style 5760 removed, 437 written.

### 1a. The large spans, broken down (snapshot line ranges)

**DRAFT-v2**

| op | block (label / heading / float) | lines | words |
|---|---|---|---|
| 5 | section{The same properties on a maze} `sec:maze` | 926–941 | 148 |
| 5 | subsection{Deletion transfers (CMAZE-2, banked)} `sec:maze-del` | 943–952 | 86 |
| 5 | `tab:maze-del` | 954–992 | 295 |
| 5 | subsection{Fixing the request layer (CMAZE-2b)} `sec:req-layer` | 994–1011 | 181 |
| 5 | subsection{Composition on routes: the verb is the lever, and …} `sec:verb` | 1013–1023 | 126 |
| 5 | `tab:verb` | 1025–1055 | 183 |
| 5 | paragraph{The verb claim.} | 1057–1069 | 155 |
| 5 | paragraph{Why it is an interface claim and not an n-CLU claim.} | 1071–1081 | 149 |
| 5 | paragraph{The lookup floor, printed honestly.} | 1083–1096 | 161 |
| 5 | subsection{World-model rows: transition and observation gener…} `sec:maze-wm` | 1098–1107 | 89 |
| 7 | against 71 to 140 s and 8,000 gradient steps for a baseline'… | 1179–1182 | 13 |
| 7 | paragraph{CMAZE-O: the control check holds; the baseline margin is withdrawn.} | 1183–1205 | 314 |
| 13 | `tab:impl-maze` | 1669–1684 | 53 |
| 20 | section{CMAZE-2 in full (original request layer, TEST, 3 o…} `app:cmaze2` | 1985–1990 | 45 |
| 20 | `tab:cmaze2` | 1992–2025 | 279 |
| 20 | paragraph{The mechanism, isolated on the scored legs (analysis, not a row).} | 2027–2041 | 191 |
| 20 | section{CMAZE-1 proper at the harness tier} `app:cmaze1` | 2042–2043 | 6 |
| 20 | `tab:cmaze1` | 2045–2074 | 129 |
| 20 | section{The read verb in full (CMAZE-2b)} `app:verb` | 2075–2090 | 123 |
| 20 | figslot `fig_maze_readverb` | 2093–2097 | 31 |
| 20 | figslot `fig_maze_conditions` | 2098–2123 | 400 |
| 20 | figslot `fig_maze_rel_tuples` | 2125–2133 | 132 |
| 20 | `tab:req` | 2136–2158 | 105 |
| 20 | `tab:req2b` | 2160–2182 | 115 |
| 20 | paragraph{The go/no-go reference, validation, whole written training plans (a re…} | 2186–2190 | 64 |
| 20 | paragraph{The leg.} | 2192–2196 | 63 |
| 20 | paragraph{The declared criteria.} | 2198–2206 | 120 |
| 20 | paragraph{Disclosures.} | 2208–2214 | 85 |
| 20 | paragraph{Exact deletion on CMAZE-2b in full.} | 2216–2234 | 238 |
| 20 | `tab:maze-del-2b` | 2236–2268 | 219 |
| 20 | section{Maze refusal displays (retired)} `app:mazerefusal` | 2269–2285 | 200 |
| 30 | `tab:cmazeo` | 2710–2739 | 488 |

**DRAFT-v2-style**

| op | block (label / heading / float) | lines | words |
|---|---|---|---|
| 5 | section{The same properties on a maze} `sec:maze` | 921–931 | 95 |
| 5 | \item \textbf{CMAZE-1}: the generator at its own defaults; | 932–932 | 7 |
| 5 | \item \textbf{CMAZE-2}: a request layer over it that places … | 933–934 | 22 |
| 5 | \item \textbf{CMAZE-2b}: CMAZE-2 with one acceptance rule ad… | 935–936 | 20 |
| 5 | \item \textbf{CMAZE-T} and \textbf{CMAZE-O}: transition and … | 937–938 | 10 |
| 5 | subsection{Deletion transfers (CMAZE-2, banked)} `sec:maze-del` | 940–949 | 88 |
| 5 | `tab:maze-del` | 951–989 | 295 |
| 5 | subsection{Fixing the request layer (CMAZE-2b)} `sec:req-layer` | 991–1008 | 185 |
| 5 | subsection{Composition on routes: the verb is the lever, and …} `sec:verb` | 1010–1020 | 131 |
| 5 | `tab:verb` | 1022–1052 | 183 |
| 5 | paragraph{The verb claim.} | 1054–1066 | 156 |
| 5 | paragraph{Why it is an interface claim and not an n-CLU claim.} | 1068–1077 | 146 |
| 5 | paragraph{The lookup floor, printed honestly.} | 1079–1090 | 157 |
| 5 | subsection{World-model rows: transition and observation gener…} `sec:maze-wm` | 1092–1101 | 89 |
| 7 | against 71 to 140 s and 8,000 gradient steps for a baseline'… | 1173–1176 | 13 |
| 7 | paragraph{CMAZE-O: the control check holds; the baseline margin is withdrawn.} | 1177–1199 | 314 |
| 13 | `tab:impl-maze` | 1657–1672 | 53 |
| 20 | section{CMAZE-2 in full (original request layer, TEST, 3 o…} `app:cmaze2` | 1973–1978 | 45 |
| 20 | `tab:cmaze2` | 1980–2013 | 282 |
| 20 | paragraph{The mechanism, isolated on the scored legs (analysis, not a row).} | 2015–2029 | 201 |
| 20 | section{CMAZE-1 proper at the harness tier} `app:cmaze1` | 2030–2031 | 6 |
| 20 | `tab:cmaze1` | 2033–2062 | 127 |
| 20 | section{The read verb in full (CMAZE-2b)} `app:verb` | 2063–2078 | 126 |
| 20 | figslot `fig_maze_readverb` | 2081–2085 | 31 |
| 20 | figslot `fig_maze_conditions` | 2086–2111 | 400 |
| 20 | figslot `fig_maze_rel_tuples` | 2113–2121 | 132 |
| 20 | `tab:req` | 2124–2146 | 105 |
| 20 | `tab:req2b` | 2148–2170 | 115 |
| 20 | paragraph{The go/no-go reference, validation, whole written training plans (a re…} | 2174–2178 | 65 |
| 20 | paragraph{The leg.} | 2180–2184 | 63 |
| 20 | paragraph{The declared criteria.} | 2186–2194 | 118 |
| 20 | paragraph{Disclosures.} | 2196–2202 | 87 |
| 20 | paragraph{Exact deletion on CMAZE-2b in full.} | 2204–2222 | 238 |
| 20 | `tab:maze-del-2b` | 2224–2256 | 219 |
| 20 | section{Maze refusal displays (retired)} `app:mazerefusal` | 2257–2273 | 205 |
| 30 | `tab:cmazeo` | 2701–2730 | 488 |

## 2. Every `\ref` removed or rewritten

**DRAFT-v2**: `\ref` tokens 187 → 152. Removed (key × count): `app:cmaze2`×1, `app:verb`×3, `fig:cmazet-deletion`×3, `fig:cmazet-families`×1, `fig:maze-rel-tuples`×1, `sec:hole`×1, `sec:maze`×1, `sec:maze-del`×1, `sec:maze-wm`×2, `sec:req-layer`×1, `sec:shapes`×1, `sec:verb`×2, `tab:cmaze2`×1, `tab:cmazeo`×2, `tab:cmazet`×2, `tab:impl`×1, `tab:impl-maze`×1, `tab:maze-del`×4, `tab:maze-del-2b`×3, `tab:req`×2, `tab:req2b`×2, `tab:verb`×1. Added: `fig:cmazet-door-tint`×1, `tab:cmazet-sym`×1.
Keys referenced but not labelled after: none.

**DRAFT-v2-style**: `\ref` tokens 188 → 152. Removed (key × count): `app:cmaze2`×1, `app:verb`×3, `fig:cmazet-deletion`×3, `fig:cmazet-families`×1, `fig:maze-rel-tuples`×1, `sec:hole`×1, `sec:maze`×1, `sec:maze-del`×1, `sec:maze-wm`×2, `sec:req-layer`×1, `sec:shapes`×2, `sec:verb`×2, `tab:cmaze2`×1, `tab:cmazeo`×2, `tab:cmazet`×2, `tab:impl`×1, `tab:impl-maze`×1, `tab:maze-del`×4, `tab:maze-del-2b`×3, `tab:req`×2, `tab:req2b`×2, `tab:verb`×1. Added: `fig:cmazet-door-tint`×1, `tab:cmazet-sym`×1.
Keys referenced but not labelled after: none.

## 3. `\PLACEHOLDER` and `\TODO`, before → after (document order)

**DRAFT-v2 `\PLACEHOLDER`**: 3 → 2; the surviving ones byte-identical and in order: **True**.
1. kept `w16 CMAZE-T per-arm generated $s_{t+1}$ (n-CLU, transformer, MLP, LSTM) with the engine's judge's marks, one success and one failure per arm: not buil…`
2. kept `w16 CMAZE-T per-arm generated $s_{t+1}$ on the \texttt{interaction} family, with the engine's judge's marks: not buildable from persisted records; may…`
3. **REMOVED** `w16 CMAZE-T post-deletion per-arm generated $s_{t+1}$ (n-CLU before / after downdate, each baseline as trained / retrained without): not buildable fro…`

**DRAFT-v2 `\TODO`**: 4 → 4; the surviving ones byte-identical and in order: **True**.
1. kept `CMAZE-T sentence reserved for the Head: the six-seed symmetric-input condition table (\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym}) replaces the ea…`
2. kept `no record prints the byte ratio of the composition-only n-CLU to the single design; the records give the two totals printed here and in Tab.~\ref{tab:…`
3. kept `AI use statement --- the authors to confirm which of the venue's required disclosure categories apply. This draft, the evidence pipeline and most of t…`
4. kept `anonymised code archive not yet built.`

**DRAFT-v2-style `\PLACEHOLDER`**: 3 → 2; the surviving ones byte-identical and in order: **True**.
1. kept `w16 CMAZE-T per-arm generated $s_{t+1}$ (n-CLU, transformer, MLP, LSTM) with the engine's judge's marks, one success and one failure per arm: not buil…`
2. kept `w16 CMAZE-T per-arm generated $s_{t+1}$ on the \texttt{interaction} family, with the engine's judge's marks: not buildable from persisted records; may…`
3. **REMOVED** `w16 CMAZE-T post-deletion per-arm generated $s_{t+1}$ (n-CLU before / after downdate, each baseline as trained / retrained without): not buildable fro…`

**DRAFT-v2-style `\TODO`**: 4 → 4; the surviving ones byte-identical and in order: **True**.
1. kept `CMAZE-T sentence reserved for the Head: the six-seed symmetric-input condition table (\S\ref{sec:maze-wm}, Table~\ref{tab:cmazet-sym}) replaces the ea…`
2. kept `no record prints the byte ratio of the composition-only n-CLU to the single design; the records give the two totals printed here and in Tab.~\ref{tab:…`
3. kept `AI use statement --- the authors to confirm which of the venue's required disclosure categories apply. This draft, the evidence pipeline and most of t…`
4. kept `anonymised code archive not yet built.`

## 4. Section, figure and table index (from the .aux of each build)

<details><summary><b>DRAFT-v2, before</b>: 92 labels</summary>

| kind | number | label | page | title / caption first words |
|---|---|---|---|---|
| section | 1 | `sec:intro` | 2 | Introduction: a naive unit, and why its three properties matter |
| subsection | 1.1 | `sec:positioning` | 2 | Positioning: a naive unit, compared with naive units |
| subsection | 1.2 | `sec:why` | 3 | Why the three properties matter: world models need all three at once |
| subsection | 1.3 | `sec:contrib` | 3 | What we show, and at what thickness |
| figure | 1 | `fig:task-shapes` | 4 | The Shapes3D task. (a) The six factors (floor hue, wall hue, object hue, scale, … |
| section | 2 | `sec:object` | 4 | The object, the judge and the memory footprint |
| figure | 2 | `fig:task-cmazet` | 5 | The CMAZE-T task. (a) One generated maze, drawn by the engine: walls, floor tint… |
| section | 3 | `sec:shapes` | 6 | The n-CLU on Shapes3D: the single design at $\Kstar $ |
| table | 1 | `tab:t7` | 7 | Composition at $\Kstar =32$ against data-matched baselines, Shapes3D value hold-… |
| subsection | 3.1 | `sec:shapes-t7` | 7 | Composition at $\Kstar $: rows 3 and 4, three scored seeds |
| table | 2 | `tab:ledger-kstar` | 8 | The footprint breakdown of the $\Kstar =32$ object, per seed, never netted. Cova… |
| subsection | 3.2 | `sec:ladder` | 8 | The $n$-way ladder at $\Kstar $, with its inference note |
| table | 3 | `tab:ladder` | 9 | The ladder at $\Kstar =32$, $n=3$ seeds, distinct valid novel tuples per combina… |
| subsection | 3.3 | `sec:shapes-del` | 9 | Exact deletion on the single design: value, conjunction, item |
| figure | 3 | `fig:ladder` | 10 | The ladder at $\Kstar $, one panel per baseline set, 3 seeds; every panel title … |
| table | 4 | `tab:del` | 11 | Deletion on the single design, the four measurements, $\Kstar =32$, rows 3--4, T… |
| subsection | 3.4 | `sec:del-comparators` | 11 | What the baselines can do instead: retrain without, or approximate |
| table | 5 | `tab:hole` | 12 | Refusal DISPLAY rows. A display is never a result. Our side is a cost line on th… |
| subsection | 3.5 | `sec:hole` | 12 | Refusal after deletion: the hole, a separate mechanism |
| subsection | 3.6 | `sec:ablations` | 12 | Two ablations, each keeping one property |
| subsection | 3.7 | `sec:adapt` | 13 | The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-C… |
| subsection | 3.7 | `sec:bard` | 13 | The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-C… |
| section | 4 | `sec:maze` | 13 | The same properties on a maze |
| subsection | 4.1 | `sec:maze-del` | 13 | Deletion transfers (CMAZE-2, banked) |
| table | 6 | `tab:maze-del` | 14 | Exact deletion on the maze, CMAZE-2, TEST, 3 of 3 scored seeds. Delete the green… |
| subsection | 4.2 | `sec:req-layer` | 14 | Fixing the request layer (CMAZE-2b) |
| subsection | 4.3 | `sec:verb` | 14 | Composition on routes: the verb is the lever, and it belongs to the interface |
| table | 7 | `tab:verb` | 15 | The read verb on CMAZE-2b, TEST, $n=3$ scored seeds. Validity under the engine's… |
| subsection | 4.4 | `sec:maze-wm` | 15 | World-model rows: transition and observation generation |
| section | 5 | `sec:limits` | 17 | Limitations of the naive unit, and a leaving note |
| subsection | 5.1 | `sec:limits-list` | 17 | What the naive unit does not do (measured, not hedged) |
| subsection | 5.2 | `sec:clue` | 18 | The clue: the n-CLU is the zero-dynamics limit of a settling unit |
| subsection | 5.3 | `sec:theory` | 18 | The theory we carry: the Latent Dynamics Model class |
| subsection | 5.4 | `sec:leaving` | 19 | Leaving note |
| section | 6 | `sec:followup` | 19 | Follow-up: a full evaluation of the physics CLU |
| appendix | A | `app:conventions` | 24 | Reporting conventions |
| appendix | B | `app:t7more` | 24 | Tab.~\ref {tab:t7}: what it does and does not claim |
| table | 8 | `tab:gentable-kstar` | 24 | The generation table, $\Kstar $ single design, one declared evaluation run, gene… |
| table | 9 | `tab:lace-poe` | 25 | LACE and PoE at their own settings, Shapes3D value hold-out, TEST, $n=3$ seeds (… |
| appendix | C | `app:k5` | 25 | The $K=5$ endpoint of the sweep |
| table | 10 | `tab:k5` | 25 | Ladder at $K=5$, the sweep's endpoint, rebuilt n-CLU, 3 seeds. n-CLU fitted on t… |
| appendix | D | `app:kcurve` | 26 | The $K$-curve: composition and the memory footprint against $K$ |
| table | 11 | `tab:kcurve` | 26 | The $K$-curve, validation, symmetric-triangle convention. The validation sweep l… |
| figure | 4 | `fig:crossover` | 27 | The $K$-curve, validation, 3 seeds at every $K$. (a)~Composition against $K$: th… |
| table | 12 | `tab:kcurve3` | 27 | The $K$-curve at $n=3$ seeds on every row, validation (a smoke, not a claim). By… |
| appendix | E | `app:impl` | 27 | Exact deletion: three implementations, and the failure they agree on |
| table | 13 | `tab:impl-maze` | 27 | Exact (n-CLU-scoped) deletion, implementation 1: the maze n-CLU (CMAZE-2). The t… |
| table | 14 | `tab:impl` | 28 | Exact (n-CLU-scoped) deletion on Shapes3D, implementations 2 and 3, bands not sh… |
| appendix | F | `app:w5` | 28 | The earlier composition cell: byte and parameter match, and the ceiling ladder |
| table | 15 | `tab:ablations` | 29 | The two ablations, earlier composition cell, TEST, $n=3$ seeds, one pin (524,288… |
| appendix | G | `app:delbase` | 29 | Deletion comparators in full (fit pool, 0 test runs) |
| table | 16 | `tab:rivals-del` | 29 | Deletion cost for the data-matched baselines: retrain without the rows, on the n… |
| table | 17 | `tab:rivals-del-seeds` | 29 | The n-CLU after deletion against each baseline retrained without the same rows, … |
| figure | 5 | `fig:deletion` | 30 | Deletion on the single design at $\Kstar $, the four measurements, rows 3--4, TE… |
| appendix | H | `app:adapt` | 30 | Light encoder adaptation in full |
| table | 18 | `tab:adapt` | 31 | Encoder adaptation at $\Kstar $, TEST, 3 seeds $\times $ 3 configurations (nine … |
| figure | 6 | `fig:adapt` | 31 | Encoder adaptation margins. Single design minus the best learned baseline per ar… |
| appendix | I | `app:encloop` | 31 | The encoder loop in full (shortened-decoder instrument, and the longer leg) |
| table | 19 | `tab:encloop` | 32 | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one de… |
| table | 20 | `tab:bard-gpu` | 32 | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 t… |
| appendix | J | `app:cmaze2` | 32 | CMAZE-2 in full (original request layer, TEST, 3 of 3 scored seeds) |
| table | 21 | `tab:cmaze2` | 32 | CMAZE-2 result rows, validity under the engine's judge, mean (seed 0 / 1 / 2). N… |
| appendix | K | `app:cmaze1` | 33 | CMAZE-1 proper at the harness tier |
| table | 22 | `tab:cmaze1` | 33 | CMAZE-1 proper, harness tier, 3 seeds, 200 evaluation instances per seed, baseli… |
| appendix | L | `app:verb` | 33 | The read verb in full (CMAZE-2b) |
| table | 23 | `tab:req` | 33 | CMAZE-2 request-layer audit (the original request layer, before the one acceptan… |
| figure | 7 | `fig:verb` | 34 | The read verb on CMAZE-2b. \texttt {AVG} against \texttt {MERGE} per arm and see… |
| table | 24 | `tab:req2b` | 34 | CMAZE-2b request-layer audit (after the one acceptance rule), seeds 0 / 1 / 2. V… |
| table | 25 | `tab:maze-del-2b` | 35 | Exact deletion on the maze, CMAZE-2b, 3 seeds, 0 test runs. This table contains … |
| appendix | M | `app:mazerefusal` | 35 | Maze refusal displays (retired) |
| appendix | N | `app:clue` | 35 | The zero-dynamics identity: conditions and provenance |
| appendix | O | `app:physics` | 36 | The retention law on the field-mode n-CLU in full |
| table | 26 | `tab:crr` | 36 | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out comb… |
| appendix | P | `app:shorts` | 37 | Prior derivations |
| appendix | Q | `app:bench` | 38 | What an evaluation must contain: the ten requirements |
| appendix | R | `app:mazewm` | 39 | Maze world-model rows |
| table | 27 | `tab:cmazet` | 40 | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed… |
| table | 27 | `tab:cmaze-del` | 40 | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed… |
| table | 28 | `tab:cmazet-sym` | 41 | CMAZE-T with symmetric inputs: every condition, unseen $n$-way, the engine's jud… |
| appendix | S | `app:encoder-main` | 41 | The encoder: adaptation and the unfreezing decomposition |
| table | 29 | `tab:cmazeo` | 42 | Maze world-model rows: CMAZE-O (observation generation), TEST, field design, 3 s… |
| appendix | T | `app:laws` | 42 | The two laws on this study's n-CLU, stated in full |
| appendix | U | `app:morelimits` | 43 | Further limitations |
| appendix | V | `app:placeholders` | 43 | Placeholder index |
| appendix | W | `app:notrun` | 43 | Declared not-runs |
| figure | 8 | `fig:maze-wm` | 45 | CMAZE-T, one row per input condition, unseen $n$-way; (a)~3 seeds per row, (b), … |
| figure | 9 | `fig:maze-rel-tuples` | 46 | CMAZE-T, n-CLU-REL, words plus observation, TEST: every held-out $n$-way tuple, … |
| figure | 10 | `fig:cmazet-families` | 47 | Held-out CMAZE-T transitions, one per rule family, every arm (words plus observa… |
| figure | 11 | `fig:cmazet-per-arm` | 48 | Each CMAZE-T arm succeeding and failing (words plus observation, TEST, seed 0). … |
| figure | 12 | `fig:cmazet-rel-per-arm` | 49 | Each CMAZE-T arm succeeding and failing under the colour-relational request codi… |
| figure | 13 | `fig:cmazet-door-tint` | 50 | Door-then-tint (\texttt {interaction) transitions, one per move direction presen… |
| figure | 14 | `fig:cmazet-deletion` | 50 | Deletion of colour 1 (value scope), seed 0, words plus observation. On the survi… |

</details>

<details><summary><b>DRAFT-v2, after</b>: 73 labels</summary>

| kind | number | label | page | title / caption first words |
|---|---|---|---|---|
| section | 1 | `sec:intro` | 2 | Introduction: a naive unit, and why its three properties matter |
| subsection | 1.1 | `sec:positioning` | 2 | Positioning: a naive unit, compared with naive units |
| subsection | 1.2 | `sec:why` | 3 | Why the three properties matter: world models need all three at once |
| subsection | 1.3 | `sec:contrib` | 3 | What we show, and at what thickness |
| figure | 1 | `fig:task-shapes` | 4 | The Shapes3D task. (a) The six factors (floor hue, wall hue, object hue, scale, … |
| section | 2 | `sec:object` | 4 | The object, the judge and the memory footprint |
| figure | 2 | `fig:task-cmazet` | 5 | The CMAZE-T task. (a) One generated maze, drawn by the engine: walls, floor tint… |
| section | 3 | `sec:shapes` | 6 | The n-CLU on Shapes3D: the single design at $\Kstar $ |
| subsection | 3.1 | `sec:shapes-t7` | 6 | Composition at $\Kstar $: rows 3 and 4, three scored seeds |
| table | 1 | `tab:t7` | 7 | Composition at $\Kstar =32$ against data-matched baselines, Shapes3D value hold-… |
| subsection | 3.2 | `sec:ladder` | 7 | The $n$-way ladder at $\Kstar $, with its inference note |
| table | 2 | `tab:ledger-kstar` | 8 | The footprint breakdown of the $\Kstar =32$ object, per seed, never netted. Cova… |
| table | 3 | `tab:ladder` | 8 | The ladder at $\Kstar =32$, $n=3$ seeds, distinct valid novel tuples per combina… |
| subsection | 3.3 | `sec:shapes-del` | 9 | Exact deletion on the single design: value, conjunction, item |
| figure | 3 | `fig:ladder` | 10 | The ladder at $\Kstar $, one panel per baseline set, 3 seeds; every panel title … |
| table | 4 | `tab:del` | 11 | Deletion on the single design, the four measurements, $\Kstar =32$, rows 3--4, T… |
| subsection | 3.4 | `sec:del-comparators` | 11 | What the baselines can do instead: retrain without, or approximate |
| table | 5 | `tab:hole` | 12 | Refusal DISPLAY rows. A display is never a result. Our side is a cost line on th… |
| subsection | 3.5 | `sec:hole` | 12 | Refusal after deletion: the hole, a separate mechanism |
| subsection | 3.6 | `sec:ablations` | 12 | Two ablations, each keeping one property |
| subsection | 3.7 | `sec:adapt` | 13 | The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-C… |
| subsection | 3.7 | `sec:bard` | 13 | The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-C… |
| section | 4 | `sec:maze` | 13 | The same properties on a maze |
| section | 4 | `sec:maze-wm` | 13 | The same properties on a maze |
| section | 5 | `sec:limits` | 14 | Limitations of the naive unit, and a leaving note |
| subsection | 5.1 | `sec:limits-list` | 14 | What the naive unit does not do (measured, not hedged) |
| subsection | 5.2 | `sec:clue` | 15 | The clue: the n-CLU is the zero-dynamics limit of a settling unit |
| subsection | 5.3 | `sec:theory` | 15 | The theory we carry: the Latent Dynamics Model class |
| subsection | 5.4 | `sec:leaving` | 16 | Leaving note |
| section | 6 | `sec:followup` | 16 | Follow-up: a full evaluation of the physics CLU |
| appendix | A | `app:conventions` | 21 | Reporting conventions |
| appendix | B | `app:t7more` | 21 | Tab.~\ref {tab:t7}: what it does and does not claim |
| table | 6 | `tab:gentable-kstar` | 21 | The generation table, $\Kstar $ single design, one declared evaluation run, gene… |
| table | 7 | `tab:lace-poe` | 22 | LACE and PoE at their own settings, Shapes3D value hold-out, TEST, $n=3$ seeds (… |
| appendix | C | `app:k5` | 22 | The $K=5$ endpoint of the sweep |
| table | 8 | `tab:k5` | 22 | Ladder at $K=5$, the sweep's endpoint, rebuilt n-CLU, 3 seeds. n-CLU fitted on t… |
| appendix | D | `app:kcurve` | 23 | The $K$-curve: composition and the memory footprint against $K$ |
| table | 9 | `tab:kcurve` | 23 | The $K$-curve, validation, symmetric-triangle convention. The validation sweep l… |
| figure | 4 | `fig:crossover` | 24 | The $K$-curve, validation, 3 seeds at every $K$. (a)~Composition against $K$: th… |
| table | 10 | `tab:kcurve3` | 24 | The $K$-curve at $n=3$ seeds on every row, validation (a smoke, not a claim). By… |
| appendix | E | `app:impl` | 24 | Deletion on Shapes3D: two implementations, and the failure they agree on |
| table | 11 | `tab:impl` | 25 | Exact (n-CLU-scoped) deletion on Shapes3D, implementations 1 and 2, bands not sh… |
| appendix | F | `app:w5` | 25 | The earlier composition cell: byte and parameter match, and the ceiling ladder |
| table | 12 | `tab:ablations` | 25 | The two ablations, earlier composition cell, TEST, $n=3$ seeds, one pin (524,288… |
| appendix | G | `app:delbase` | 26 | Deletion comparators in full (fit pool, 0 test runs) |
| table | 13 | `tab:rivals-del` | 26 | Deletion cost for the data-matched baselines: retrain without the rows, on the n… |
| table | 14 | `tab:rivals-del-seeds` | 26 | The n-CLU after deletion against each baseline retrained without the same rows, … |
| appendix | H | `app:adapt` | 26 | Light encoder adaptation in full |
| figure | 5 | `fig:deletion` | 27 | Deletion on the single design at $\Kstar $, the four measurements, rows 3--4, TE… |
| appendix | I | `app:encloop` | 27 | The encoder loop in full (shortened-decoder instrument, and the longer leg) |
| table | 15 | `tab:adapt` | 28 | Encoder adaptation at $\Kstar $, TEST, 3 seeds $\times $ 3 configurations (nine … |
| figure | 6 | `fig:adapt` | 28 | Encoder adaptation margins. Single design minus the best learned baseline per ar… |
| appendix | J | `app:clue` | 28 | The zero-dynamics identity: conditions and provenance |
| table | 16 | `tab:encloop` | 29 | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one de… |
| table | 17 | `tab:bard-gpu` | 29 | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 t… |
| table | 18 | `tab:crr` | 30 | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out comb… |
| appendix | K | `app:physics` | 30 | The retention law on the field-mode n-CLU in full |
| appendix | L | `app:shorts` | 30 | Prior derivations |
| appendix | M | `app:bench` | 32 | What an evaluation must contain: the ten requirements |
| appendix | N | `app:mazewm` | 32 | Maze world-model rows |
| table | 19 | `tab:cmazet` | 33 | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed… |
| table | 19 | `tab:cmaze-del` | 33 | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed… |
| table | 20 | `tab:cmazet-sym` | 34 | CMAZE-T with symmetric inputs: every condition, unseen $n$-way, the engine's jud… |
| appendix | O | `app:encoder-main` | 34 | The encoder: adaptation and the unfreezing decomposition |
| appendix | P | `app:laws` | 35 | The two laws on this study's n-CLU, stated in full |
| appendix | Q | `app:morelimits` | 35 | Further limitations |
| appendix | R | `app:placeholders` | 36 | Placeholder index |
| appendix | S | `app:notrun` | 36 | Declared not-runs |
| figure | 7 | `fig:maze-wm` | 37 | CMAZE-T, one row per input condition, unseen $n$-way; (a)~3 seeds per row, (b), … |
| figure | 8 | `fig:cmazet-families` | 38 | Held-out CMAZE-T transitions, one per rule family, every arm (words plus observa… |
| figure | 9 | `fig:cmazet-per-arm` | 39 | Each CMAZE-T arm succeeding and failing (words plus observation, TEST, seed 0). … |
| figure | 10 | `fig:cmazet-rel-per-arm` | 40 | Each CMAZE-T arm succeeding and failing under the colour-relational request codi… |
| figure | 11 | `fig:cmazet-door-tint` | 41 | Door-then-tint (\texttt {interaction) transitions, one per move direction presen… |

</details>

<details><summary><b>DRAFT-v2-style, before</b>: 92 labels</summary>

| kind | number | label | page | title / caption first words |
|---|---|---|---|---|
| section | 1 | `sec:intro` | 2 | Introduction: a naive unit, and why its three properties matter |
| subsection | 1.1 | `sec:positioning` | 2 | Positioning: a naive unit, compared with naive units |
| subsection | 1.2 | `sec:why` | 3 | Why the three properties matter: world models need all three at once |
| subsection | 1.3 | `sec:contrib` | 3 | What we show, and at what thickness |
| figure | 1 | `fig:task-shapes` | 4 | The Shapes3D task. (a) The six factors (floor hue, wall hue, object hue, scale, … |
| section | 2 | `sec:object` | 4 | The object, the judge and the memory footprint |
| figure | 2 | `fig:task-cmazet` | 5 | The CMAZE-T task. (a) One generated maze, drawn by the engine: walls, floor tint… |
| section | 3 | `sec:shapes` | 6 | The n-CLU on Shapes3D: the single design at $\Kstar $ |
| table | 1 | `tab:t7` | 7 | Composition at $\Kstar =32$ against data-matched baselines, Shapes3D value hold-… |
| subsection | 3.1 | `sec:shapes-t7` | 7 | Composition at $\Kstar $: rows 3 and 4, three scored seeds |
| table | 2 | `tab:ledger-kstar` | 8 | The footprint breakdown of the $\Kstar =32$ object, per seed, never netted. Cova… |
| subsection | 3.2 | `sec:ladder` | 8 | The $n$-way ladder at $\Kstar $, with its inference note |
| table | 3 | `tab:ladder` | 9 | The ladder at $\Kstar =32$, $n=3$ seeds, distinct valid novel tuples per combina… |
| figure | 3 | `fig:ladder` | 10 | The ladder at $\Kstar $, one panel per baseline set, 3 seeds; every panel title … |
| subsection | 3.3 | `sec:shapes-del` | 10 | Exact deletion on the single design: value, conjunction, item |
| table | 4 | `tab:del` | 11 | Deletion on the single design, the four measurements, $\Kstar =32$, rows 3--4, T… |
| table | 5 | `tab:hole` | 12 | Refusal DISPLAY rows. A display is never a result. Our side is a cost line on th… |
| subsection | 3.4 | `sec:del-comparators` | 12 | What the baselines can do instead: retrain without, or approximate |
| subsection | 3.5 | `sec:hole` | 12 | Refusal after deletion: the hole, a separate mechanism |
| subsection | 3.6 | `sec:ablations` | 13 | Two ablations, each keeping one property |
| subsection | 3.7 | `sec:adapt` | 13 | The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-C… |
| subsection | 3.7 | `sec:bard` | 13 | The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-C… |
| section | 4 | `sec:maze` | 13 | The same properties on a maze |
| table | 6 | `tab:maze-del` | 14 | Exact deletion on the maze, CMAZE-2, TEST, 3 of 3 scored seeds. Delete the green… |
| subsection | 4.1 | `sec:maze-del` | 14 | Deletion transfers (CMAZE-2, banked) |
| subsection | 4.2 | `sec:req-layer` | 14 | Fixing the request layer (CMAZE-2b) |
| table | 7 | `tab:verb` | 15 | The read verb on CMAZE-2b, TEST, $n=3$ scored seeds. Validity under the engine's… |
| subsection | 4.3 | `sec:verb` | 15 | Composition on routes: the verb is the lever, and it belongs to the interface |
| subsection | 4.4 | `sec:maze-wm` | 16 | World-model rows: transition and observation generation |
| section | 5 | `sec:limits` | 17 | Limitations of the naive unit, and a leaving note |
| subsection | 5.1 | `sec:limits-list` | 17 | What the naive unit does not do (measured, not hedged) |
| subsection | 5.2 | `sec:clue` | 18 | The clue: the n-CLU is the zero-dynamics limit of a settling unit |
| subsection | 5.3 | `sec:theory` | 19 | The theory we carry: the Latent Dynamics Model class |
| subsection | 5.4 | `sec:leaving` | 19 | Leaving note |
| section | 6 | `sec:followup` | 19 | Follow-up: a full evaluation of the physics CLU |
| appendix | A | `app:conventions` | 24 | Reporting conventions |
| appendix | B | `app:t7more` | 24 | Tab.~\ref {tab:t7}: what it does and does not claim |
| table | 8 | `tab:gentable-kstar` | 25 | The generation table, $\Kstar $ single design, one declared evaluation run, gene… |
| table | 9 | `tab:lace-poe` | 25 | LACE and PoE at their own settings, Shapes3D value hold-out, TEST, $n=3$ seeds (… |
| appendix | C | `app:k5` | 25 | The $K=5$ endpoint of the sweep |
| table | 10 | `tab:k5` | 26 | Ladder at $K=5$, the sweep's endpoint, rebuilt n-CLU, 3 seeds. n-CLU fitted on t… |
| appendix | D | `app:kcurve` | 26 | The $K$-curve: composition and the memory footprint against $K$ |
| table | 11 | `tab:kcurve` | 26 | The $K$-curve, validation, symmetric-triangle convention. The validation sweep l… |
| figure | 4 | `fig:crossover` | 27 | The $K$-curve, validation, 3 seeds at every $K$. (a)~Composition against $K$: th… |
| table | 12 | `tab:kcurve3` | 27 | The $K$-curve at $n=3$ seeds on every row, validation (a smoke, not a claim). By… |
| appendix | E | `app:impl` | 27 | Exact deletion: three implementations, and the failure they agree on |
| table | 13 | `tab:impl-maze` | 28 | Exact (n-CLU-scoped) deletion, implementation 1: the maze n-CLU (CMAZE-2). The t… |
| table | 14 | `tab:impl` | 28 | Exact (n-CLU-scoped) deletion on Shapes3D, implementations 2 and 3, bands not sh… |
| table | 15 | `tab:ablations` | 29 | The two ablations, earlier composition cell, TEST, $n=3$ seeds, one pin (524,288… |
| appendix | F | `app:w5` | 29 | The earlier composition cell: byte and parameter match, and the ceiling ladder |
| appendix | G | `app:delbase` | 29 | Deletion comparators in full (fit pool, 0 test runs) |
| table | 16 | `tab:rivals-del` | 29 | Deletion cost for the data-matched baselines: retrain without the rows, on the n… |
| appendix | H | `app:adapt` | 29 | Light encoder adaptation in full |
| figure | 5 | `fig:deletion` | 30 | Deletion on the single design at $\Kstar $, the four measurements, rows 3--4, TE… |
| appendix | I | `app:encloop` | 30 | The encoder loop in full (shortened-decoder instrument, and the longer leg) |
| table | 17 | `tab:rivals-del-seeds` | 31 | The n-CLU after deletion against each baseline retrained without the same rows, … |
| table | 18 | `tab:adapt` | 31 | Encoder adaptation at $\Kstar $, TEST, 3 seeds $\times $ 3 configurations (nine … |
| appendix | J | `app:cmaze2` | 31 | CMAZE-2 in full (original request layer, TEST, 3 of 3 scored seeds) |
| figure | 6 | `fig:adapt` | 32 | Encoder adaptation margins. Single design minus the best learned baseline per ar… |
| table | 19 | `tab:encloop` | 32 | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one de… |
| table | 20 | `tab:bard-gpu` | 33 | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 t… |
| table | 21 | `tab:cmaze2` | 33 | CMAZE-2 result rows, validity under the engine's judge, mean (seed 0 / 1 / 2). N… |
| appendix | K | `app:cmaze1` | 33 | CMAZE-1 proper at the harness tier |
| table | 22 | `tab:cmaze1` | 33 | CMAZE-1 proper, harness tier, 3 seeds, 200 evaluation instances per seed, baseli… |
| appendix | L | `app:verb` | 33 | The read verb in full (CMAZE-2b) |
| figure | 7 | `fig:verb` | 34 | The read verb on CMAZE-2b. \texttt {AVG} against \texttt {MERGE} per arm and see… |
| table | 23 | `tab:req` | 34 | CMAZE-2 request-layer audit (the original request layer, before the one acceptan… |
| table | 24 | `tab:req2b` | 35 | CMAZE-2b request-layer audit (after the one acceptance rule), seeds 0 / 1 / 2. V… |
| table | 25 | `tab:maze-del-2b` | 35 | Exact deletion on the maze, CMAZE-2b, 3 seeds, 0 test runs. This table contains … |
| appendix | M | `app:mazerefusal` | 35 | Maze refusal displays (retired) |
| appendix | N | `app:clue` | 36 | The zero-dynamics identity: conditions and provenance |
| appendix | O | `app:physics` | 36 | The retention law on the field-mode n-CLU in full |
| table | 26 | `tab:crr` | 37 | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out comb… |
| appendix | P | `app:shorts` | 37 | Prior derivations |
| appendix | Q | `app:bench` | 39 | What an evaluation must contain: the ten requirements |
| appendix | R | `app:mazewm` | 39 | Maze world-model rows |
| table | 27 | `tab:cmazet` | 40 | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed… |
| table | 27 | `tab:cmaze-del` | 40 | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed… |
| table | 28 | `tab:cmazet-sym` | 41 | CMAZE-T with symmetric inputs: every condition, unseen $n$-way, the engine's jud… |
| table | 29 | `tab:cmazeo` | 42 | Maze world-model rows: CMAZE-O (observation generation), TEST, field design, 3 s… |
| appendix | S | `app:encoder-main` | 42 | The encoder: adaptation and the unfreezing decomposition |
| appendix | T | `app:laws` | 43 | The two laws on this study's n-CLU, stated in full |
| appendix | U | `app:morelimits` | 43 | Further limitations |
| appendix | V | `app:placeholders` | 44 | Placeholder index |
| appendix | W | `app:notrun` | 44 | Declared not-runs |
| figure | 8 | `fig:maze-wm` | 45 | CMAZE-T, one row per input condition, unseen $n$-way; (a)~3 seeds per row, (b), … |
| figure | 9 | `fig:maze-rel-tuples` | 46 | CMAZE-T, n-CLU-REL, words plus observation, TEST: every held-out $n$-way tuple, … |
| figure | 10 | `fig:cmazet-families` | 47 | Held-out CMAZE-T transitions, one per rule family, every arm (words plus observa… |
| figure | 11 | `fig:cmazet-per-arm` | 48 | Each CMAZE-T arm succeeding and failing (words plus observation, TEST, seed 0). … |
| figure | 12 | `fig:cmazet-rel-per-arm` | 49 | Each CMAZE-T arm succeeding and failing under the colour-relational request codi… |
| figure | 13 | `fig:cmazet-door-tint` | 50 | Door-then-tint (\texttt {interaction) transitions, one per move direction presen… |
| figure | 14 | `fig:cmazet-deletion` | 50 | Deletion of colour 1 (value scope), seed 0, words plus observation. On the survi… |

</details>

<details><summary><b>DRAFT-v2-style, after</b>: 73 labels</summary>

| kind | number | label | page | title / caption first words |
|---|---|---|---|---|
| section | 1 | `sec:intro` | 2 | Introduction: a naive unit, and why its three properties matter |
| subsection | 1.1 | `sec:positioning` | 2 | Positioning: a naive unit, compared with naive units |
| subsection | 1.2 | `sec:why` | 3 | Why the three properties matter: world models need all three at once |
| subsection | 1.3 | `sec:contrib` | 3 | What we show, and at what thickness |
| figure | 1 | `fig:task-shapes` | 4 | The Shapes3D task. (a) The six factors (floor hue, wall hue, object hue, scale, … |
| section | 2 | `sec:object` | 4 | The object, the judge and the memory footprint |
| figure | 2 | `fig:task-cmazet` | 5 | The CMAZE-T task. (a) One generated maze, drawn by the engine: walls, floor tint… |
| section | 3 | `sec:shapes` | 6 | The n-CLU on Shapes3D: the single design at $\Kstar $ |
| table | 1 | `tab:t7` | 7 | Composition at $\Kstar =32$ against data-matched baselines, Shapes3D value hold-… |
| subsection | 3.1 | `sec:shapes-t7` | 7 | Composition at $\Kstar $: rows 3 and 4, three scored seeds |
| subsection | 3.2 | `sec:ladder` | 7 | The $n$-way ladder at $\Kstar $, with its inference note |
| table | 2 | `tab:ledger-kstar` | 8 | The footprint breakdown of the $\Kstar =32$ object, per seed, never netted. Cova… |
| table | 3 | `tab:ladder` | 8 | The ladder at $\Kstar =32$, $n=3$ seeds, distinct valid novel tuples per combina… |
| subsection | 3.3 | `sec:shapes-del` | 9 | Exact deletion on the single design: value, conjunction, item |
| figure | 3 | `fig:ladder` | 10 | The ladder at $\Kstar $, one panel per baseline set, 3 seeds; every panel title … |
| table | 4 | `tab:del` | 11 | Deletion on the single design, the four measurements, $\Kstar =32$, rows 3--4, T… |
| subsection | 3.4 | `sec:del-comparators` | 11 | What the baselines can do instead: retrain without, or approximate |
| table | 5 | `tab:hole` | 12 | Refusal DISPLAY rows. A display is never a result. Our side is a cost line on th… |
| subsection | 3.5 | `sec:hole` | 12 | Refusal after deletion: the hole, a separate mechanism |
| subsection | 3.6 | `sec:ablations` | 12 | Two ablations, each keeping one property |
| subsection | 3.7 | `sec:adapt` | 13 | The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-C… |
| subsection | 3.7 | `sec:bard` | 13 | The encoder: frozen by design, lightly adapted; the encoder loop licenses no n-C… |
| section | 4 | `sec:maze` | 13 | The same properties on a maze |
| section | 4 | `sec:maze-wm` | 13 | The same properties on a maze |
| section | 5 | `sec:limits` | 14 | Limitations of the naive unit, and a leaving note |
| subsection | 5.1 | `sec:limits-list` | 15 | What the naive unit does not do (measured, not hedged) |
| subsection | 5.2 | `sec:clue` | 15 | The clue: the n-CLU is the zero-dynamics limit of a settling unit |
| subsection | 5.3 | `sec:theory` | 16 | The theory we carry: the Latent Dynamics Model class |
| subsection | 5.4 | `sec:leaving` | 16 | Leaving note |
| section | 6 | `sec:followup` | 16 | Follow-up: a full evaluation of the physics CLU |
| appendix | A | `app:conventions` | 21 | Reporting conventions |
| appendix | B | `app:t7more` | 21 | Tab.~\ref {tab:t7}: what it does and does not claim |
| table | 6 | `tab:gentable-kstar` | 22 | The generation table, $\Kstar $ single design, one declared evaluation run, gene… |
| table | 7 | `tab:lace-poe` | 22 | LACE and PoE at their own settings, Shapes3D value hold-out, TEST, $n=3$ seeds (… |
| appendix | C | `app:k5` | 22 | The $K=5$ endpoint of the sweep |
| table | 8 | `tab:k5` | 23 | Ladder at $K=5$, the sweep's endpoint, rebuilt n-CLU, 3 seeds. n-CLU fitted on t… |
| appendix | D | `app:kcurve` | 23 | The $K$-curve: composition and the memory footprint against $K$ |
| table | 9 | `tab:kcurve` | 23 | The $K$-curve, validation, symmetric-triangle convention. The validation sweep l… |
| figure | 4 | `fig:crossover` | 24 | The $K$-curve, validation, 3 seeds at every $K$. (a)~Composition against $K$: th… |
| table | 10 | `tab:kcurve3` | 24 | The $K$-curve at $n=3$ seeds on every row, validation (a smoke, not a claim). By… |
| appendix | E | `app:impl` | 24 | Deletion on Shapes3D: two implementations, and the failure they agree on |
| table | 11 | `tab:impl` | 25 | Exact (n-CLU-scoped) deletion on Shapes3D, implementations 1 and 2, bands not sh… |
| appendix | F | `app:w5` | 25 | The earlier composition cell: byte and parameter match, and the ceiling ladder |
| table | 12 | `tab:ablations` | 26 | The two ablations, earlier composition cell, TEST, $n=3$ seeds, one pin (524,288… |
| appendix | G | `app:delbase` | 26 | Deletion comparators in full (fit pool, 0 test runs) |
| table | 13 | `tab:rivals-del` | 26 | Deletion cost for the data-matched baselines: retrain without the rows, on the n… |
| table | 14 | `tab:rivals-del-seeds` | 26 | The n-CLU after deletion against each baseline retrained without the same rows, … |
| appendix | H | `app:adapt` | 26 | Light encoder adaptation in full |
| figure | 5 | `fig:deletion` | 27 | Deletion on the single design at $\Kstar $, the four measurements, rows 3--4, TE… |
| table | 15 | `tab:adapt` | 28 | Encoder adaptation at $\Kstar $, TEST, 3 seeds $\times $ 3 configurations (nine … |
| figure | 6 | `fig:adapt` | 28 | Encoder adaptation margins. Single design minus the best learned baseline per ar… |
| appendix | I | `app:encloop` | 28 | The encoder loop in full (shortened-decoder instrument, and the longer leg) |
| table | 16 | `tab:encloop` | 29 | Encoder loop, shortened decoder (400 joint steps, 8,192 images), 3 seeds, one de… |
| table | 17 | `tab:bard-gpu` | 29 | The unfreezing decomposition on the longer leg only (4,000 joint steps, 32,768 t… |
| appendix | J | `app:clue` | 29 | The zero-dynamics identity: conditions and provenance |
| appendix | K | `app:physics` | 30 | The retention law on the field-mode n-CLU in full |
| table | 18 | `tab:crr` | 30 | Declared criteria of the retention law on the field-mode n-CLU, 49 held-out comb… |
| appendix | L | `app:shorts` | 31 | Prior derivations |
| appendix | M | `app:bench` | 32 | What an evaluation must contain: the ten requirements |
| appendix | N | `app:mazewm` | 32 | Maze world-model rows |
| table | 19 | `tab:cmazet` | 33 | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed… |
| table | 19 | `tab:cmaze-del` | 33 | Maze world-model rows: CMAZE-T (transition generation), TEST, base n-CLU, 3 seed… |
| table | 20 | `tab:cmazet-sym` | 34 | CMAZE-T with symmetric inputs: every condition, unseen $n$-way, the engine's jud… |
| appendix | O | `app:encoder-main` | 34 | The encoder: adaptation and the unfreezing decomposition |
| appendix | P | `app:laws` | 35 | The two laws on this study's n-CLU, stated in full |
| appendix | Q | `app:morelimits` | 36 | Further limitations |
| appendix | R | `app:placeholders` | 36 | Placeholder index |
| appendix | S | `app:notrun` | 36 | Declared not-runs |
| figure | 7 | `fig:maze-wm` | 38 | CMAZE-T, one row per input condition, unseen $n$-way; (a)~3 seeds per row, (b), … |
| figure | 8 | `fig:cmazet-families` | 39 | Held-out CMAZE-T transitions, one per rule family, every arm (words plus observa… |
| figure | 9 | `fig:cmazet-per-arm` | 40 | Each CMAZE-T arm succeeding and failing (words plus observation, TEST, seed 0). … |
| figure | 10 | `fig:cmazet-rel-per-arm` | 41 | Each CMAZE-T arm succeeding and failing under the colour-relational request codi… |
| figure | 11 | `fig:cmazet-door-tint` | 42 | Door-then-tint (\texttt {interaction) transitions, one per move direction presen… |

</details>

### 4a. Figure and table numbers, before → after (DRAFT-v2)

| label | before (number, page) | after (number, page) |
|---|---|---|
| `fig:task-shapes` | Fig./Tab. 1, p. 4 | Fig./Tab. 1, p. 4 |
| `fig:task-cmazet` | Fig./Tab. 2, p. 5 | Fig./Tab. 2, p. 5 |
| `fig:ladder` | Fig./Tab. 3, p. 10 | Fig./Tab. 3, p. 10 |
| `fig:crossover` | Fig./Tab. 4, p. 27 | Fig./Tab. 4, p. 24 |
| `fig:deletion` | Fig./Tab. 5, p. 30 | Fig./Tab. 5, p. 27 |
| `fig:adapt` | Fig./Tab. 6, p. 31 | Fig./Tab. 6, p. 28 |
| `fig:verb` | Fig./Tab. 7, p. 34 | **removed** |
| `fig:maze-wm` | Fig./Tab. 8, p. 45 | Fig./Tab. 7, p. 37 |
| `fig:maze-rel-tuples` | Fig./Tab. 9, p. 46 | **removed** |
| `fig:cmazet-families` | Fig./Tab. 10, p. 47 | Fig./Tab. 8, p. 38 |
| `fig:cmazet-per-arm` | Fig./Tab. 11, p. 48 | Fig./Tab. 9, p. 39 |
| `fig:cmazet-rel-per-arm` | Fig./Tab. 12, p. 49 | Fig./Tab. 10, p. 40 |
| `fig:cmazet-door-tint` | Fig./Tab. 13, p. 50 | Fig./Tab. 11, p. 41 |
| `fig:cmazet-deletion` | Fig./Tab. 14, p. 50 | **removed** |
| `tab:t7` | Fig./Tab. 1, p. 7 | Fig./Tab. 1, p. 7 |
| `tab:ledger-kstar` | Fig./Tab. 2, p. 8 | Fig./Tab. 2, p. 8 |
| `tab:ladder` | Fig./Tab. 3, p. 9 | Fig./Tab. 3, p. 8 |
| `tab:del` | Fig./Tab. 4, p. 11 | Fig./Tab. 4, p. 11 |
| `tab:hole` | Fig./Tab. 5, p. 12 | Fig./Tab. 5, p. 12 |
| `tab:maze-del` | Fig./Tab. 6, p. 14 | **removed** |
| `tab:verb` | Fig./Tab. 7, p. 15 | **removed** |
| `tab:gentable-kstar` | Fig./Tab. 8, p. 24 | Fig./Tab. 6, p. 21 |
| `tab:lace-poe` | Fig./Tab. 9, p. 25 | Fig./Tab. 7, p. 22 |
| `tab:k5` | Fig./Tab. 10, p. 25 | Fig./Tab. 8, p. 22 |
| `tab:kcurve` | Fig./Tab. 11, p. 26 | Fig./Tab. 9, p. 23 |
| `tab:kcurve3` | Fig./Tab. 12, p. 27 | Fig./Tab. 10, p. 24 |
| `tab:impl-maze` | Fig./Tab. 13, p. 27 | **removed** |
| `tab:impl` | Fig./Tab. 14, p. 28 | Fig./Tab. 11, p. 25 |
| `tab:ablations` | Fig./Tab. 15, p. 29 | Fig./Tab. 12, p. 25 |
| `tab:rivals-del` | Fig./Tab. 16, p. 29 | Fig./Tab. 13, p. 26 |
| `tab:rivals-del-seeds` | Fig./Tab. 17, p. 29 | Fig./Tab. 14, p. 26 |
| `tab:adapt` | Fig./Tab. 18, p. 31 | Fig./Tab. 15, p. 28 |
| `tab:encloop` | Fig./Tab. 19, p. 32 | Fig./Tab. 16, p. 29 |
| `tab:bard-gpu` | Fig./Tab. 20, p. 32 | Fig./Tab. 17, p. 29 |
| `tab:cmaze2` | Fig./Tab. 21, p. 32 | **removed** |
| `tab:cmaze1` | Fig./Tab. 22, p. 33 | **removed** |
| `tab:req` | Fig./Tab. 23, p. 33 | **removed** |
| `tab:req2b` | Fig./Tab. 24, p. 34 | **removed** |
| `tab:maze-del-2b` | Fig./Tab. 25, p. 35 | **removed** |
| `tab:crr` | Fig./Tab. 26, p. 36 | Fig./Tab. 18, p. 30 |
| `tab:cmazet` | Fig./Tab. 27, p. 40 | Fig./Tab. 19, p. 33 |
| `tab:cmaze-del` | Fig./Tab. 27, p. 40 | Fig./Tab. 19, p. 33 |
| `tab:cmazet-sym` | Fig./Tab. 28, p. 41 | Fig./Tab. 20, p. 34 |
| `tab:cmazeo` | Fig./Tab. 29, p. 42 | **removed** |

## 5. Page counts

| draft | pages before | pages after |
|---|---|---|
| DRAFT-v2 | 50 | 41 |
| DRAFT-v2-style | 50 | 42 |

**DRAFT-v2, before** (top-level sections: number, start page, span in pages up to the next top-level section; the last runs to the end):

| § | title | start p. | pages |
|---|---|---|---|
| 1 | Introduction: a naive unit, and why its three properties matter | 2 | 2 |
| 2 | The object, the judge and the memory footprint | 4 | 2 |
| 3 | The n-CLU on Shapes3D: the single design at $\Kstar $ | 6 | 7 |
| 4 | The same properties on a maze | 13 | 4 |
| 5 | Limitations of the naive unit, and a leaving note | 17 | 2 |
| 6 | Follow-up: a full evaluation of the physics CLU | 19 | 5 |
| A | Reporting conventions | 24 | 0 |
| B | Tab.~\ref {tab:t7}: what it does and does not claim | 24 | 1 |
| C | The $K=5$ endpoint of the sweep | 25 | 1 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 26 | 1 |
| E | Exact deletion: three implementations, and the failure they agree on | 27 | 1 |
| F | The earlier composition cell: byte and parameter match, and the ceilin… | 28 | 1 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 29 | 1 |
| H | Light encoder adaptation in full | 30 | 1 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer… | 31 | 1 |
| J | CMAZE-2 in full (original request layer, TEST, 3 of 3 scored seeds) | 32 | 1 |
| K | CMAZE-1 proper at the harness tier | 33 | 0 |
| L | The read verb in full (CMAZE-2b) | 33 | 2 |
| M | Maze refusal displays (retired) | 35 | 0 |
| N | The zero-dynamics identity: conditions and provenance | 35 | 1 |
| O | The retention law on the field-mode n-CLU in full | 36 | 1 |
| P | Prior derivations | 37 | 1 |
| Q | What an evaluation must contain: the ten requirements | 38 | 1 |
| R | Maze world-model rows | 39 | 2 |
| S | The encoder: adaptation and the unfreezing decomposition | 41 | 1 |
| T | The two laws on this study's n-CLU, stated in full | 42 | 1 |
| U | Further limitations | 43 | 0 |
| V | Placeholder index | 43 | 0 |
| W | Declared not-runs | 43 | 8 |

**DRAFT-v2, after** (top-level sections: number, start page, span in pages up to the next top-level section; the last runs to the end):

| § | title | start p. | pages |
|---|---|---|---|
| 1 | Introduction: a naive unit, and why its three properties matter | 2 | 2 |
| 2 | The object, the judge and the memory footprint | 4 | 2 |
| 3 | The n-CLU on Shapes3D: the single design at $\Kstar $ | 6 | 7 |
| 4 | The same properties on a maze | 13 | 0 |
| 4 | The same properties on a maze | 13 | 1 |
| 5 | Limitations of the naive unit, and a leaving note | 14 | 2 |
| 6 | Follow-up: a full evaluation of the physics CLU | 16 | 5 |
| A | Reporting conventions | 21 | 0 |
| B | Tab.~\ref {tab:t7}: what it does and does not claim | 21 | 1 |
| C | The $K=5$ endpoint of the sweep | 22 | 1 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 23 | 1 |
| E | Deletion on Shapes3D: two implementations, and the failure they agree … | 24 | 1 |
| F | The earlier composition cell: byte and parameter match, and the ceilin… | 25 | 1 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 26 | 0 |
| H | Light encoder adaptation in full | 26 | 1 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer… | 27 | 1 |
| J | The zero-dynamics identity: conditions and provenance | 28 | 2 |
| K | The retention law on the field-mode n-CLU in full | 30 | 0 |
| L | Prior derivations | 30 | 2 |
| M | What an evaluation must contain: the ten requirements | 32 | 0 |
| N | Maze world-model rows | 32 | 2 |
| O | The encoder: adaptation and the unfreezing decomposition | 34 | 1 |
| P | The two laws on this study's n-CLU, stated in full | 35 | 0 |
| Q | Further limitations | 35 | 1 |
| R | Placeholder index | 36 | 0 |
| S | Declared not-runs | 36 | 6 |

**DRAFT-v2-style, before** (top-level sections: number, start page, span in pages up to the next top-level section; the last runs to the end):

| § | title | start p. | pages |
|---|---|---|---|
| 1 | Introduction: a naive unit, and why its three properties matter | 2 | 2 |
| 2 | The object, the judge and the memory footprint | 4 | 2 |
| 3 | The n-CLU on Shapes3D: the single design at $\Kstar $ | 6 | 7 |
| 4 | The same properties on a maze | 13 | 4 |
| 5 | Limitations of the naive unit, and a leaving note | 17 | 2 |
| 6 | Follow-up: a full evaluation of the physics CLU | 19 | 5 |
| A | Reporting conventions | 24 | 0 |
| B | Tab.~\ref {tab:t7}: what it does and does not claim | 24 | 1 |
| C | The $K=5$ endpoint of the sweep | 25 | 1 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 26 | 1 |
| E | Exact deletion: three implementations, and the failure they agree on | 27 | 2 |
| F | The earlier composition cell: byte and parameter match, and the ceilin… | 29 | 0 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 29 | 0 |
| H | Light encoder adaptation in full | 29 | 1 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer… | 30 | 1 |
| J | CMAZE-2 in full (original request layer, TEST, 3 of 3 scored seeds) | 31 | 2 |
| K | CMAZE-1 proper at the harness tier | 33 | 0 |
| L | The read verb in full (CMAZE-2b) | 33 | 2 |
| M | Maze refusal displays (retired) | 35 | 1 |
| N | The zero-dynamics identity: conditions and provenance | 36 | 0 |
| O | The retention law on the field-mode n-CLU in full | 36 | 1 |
| P | Prior derivations | 37 | 2 |
| Q | What an evaluation must contain: the ten requirements | 39 | 0 |
| R | Maze world-model rows | 39 | 3 |
| S | The encoder: adaptation and the unfreezing decomposition | 42 | 1 |
| T | The two laws on this study's n-CLU, stated in full | 43 | 0 |
| U | Further limitations | 43 | 1 |
| V | Placeholder index | 44 | 0 |
| W | Declared not-runs | 44 | 7 |

**DRAFT-v2-style, after** (top-level sections: number, start page, span in pages up to the next top-level section; the last runs to the end):

| § | title | start p. | pages |
|---|---|---|---|
| 1 | Introduction: a naive unit, and why its three properties matter | 2 | 2 |
| 2 | The object, the judge and the memory footprint | 4 | 2 |
| 3 | The n-CLU on Shapes3D: the single design at $\Kstar $ | 6 | 7 |
| 4 | The same properties on a maze | 13 | 0 |
| 4 | The same properties on a maze | 13 | 1 |
| 5 | Limitations of the naive unit, and a leaving note | 14 | 2 |
| 6 | Follow-up: a full evaluation of the physics CLU | 16 | 5 |
| A | Reporting conventions | 21 | 0 |
| B | Tab.~\ref {tab:t7}: what it does and does not claim | 21 | 1 |
| C | The $K=5$ endpoint of the sweep | 22 | 1 |
| D | The $K$-curve: composition and the memory footprint against $K$ | 23 | 1 |
| E | Deletion on Shapes3D: two implementations, and the failure they agree … | 24 | 1 |
| F | The earlier composition cell: byte and parameter match, and the ceilin… | 25 | 1 |
| G | Deletion comparators in full (fit pool, 0 test runs) | 26 | 0 |
| H | Light encoder adaptation in full | 26 | 2 |
| I | The encoder loop in full (shortened-decoder instrument, and the longer… | 28 | 1 |
| J | The zero-dynamics identity: conditions and provenance | 29 | 1 |
| K | The retention law on the field-mode n-CLU in full | 30 | 1 |
| L | Prior derivations | 31 | 1 |
| M | What an evaluation must contain: the ten requirements | 32 | 0 |
| N | Maze world-model rows | 32 | 2 |
| O | The encoder: adaptation and the unfreezing decomposition | 34 | 1 |
| P | The two laws on this study's n-CLU, stated in full | 35 | 1 |
| Q | Further limitations | 36 | 0 |
| R | Placeholder index | 36 | 0 |
| S | Declared not-runs | 36 | 7 |


## Branch and bundle
Branch `pw1/draft-cuts` commit `08d0b79b870c` (tooling in `pw1_draft_cuts/`; no outputs committed). Bundle `pw1-draft-cuts.bundle`, `git bundle verify` OK.
