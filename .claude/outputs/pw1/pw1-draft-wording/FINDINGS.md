# FINDINGS — pw1-draft-wording (PW-1 plan items 7, 8, 13 in both drafts)

Session `pw1-draft-wording` · branch `pw1/draft-wording` (from `pw1/draft-figures` a01bea7) · batmac · start in START_UTC.txt.
Tooling (committed): `pw1_draft_wording/edit.py` (56 anchored operations per draft; each operation lists one alternative per draft wording, and the first anchor that occurs exactly once is applied), `report.py` (before→after table), `check_stage.sh` (port of pw1_draft_figures/check_stage.sh with the hard-coded worktree path removed). Reused: `pw1_draft_sweeps/{compile_check.sh,style_diff_check.py,check_provenance.py,scan.py,texnum.py}`, `pw1_draft_figures/captions.py`, `pw1_draft_cuts/pdftext.py`.
Evidence here: `snap-before/` (live drafts and PDFs before, hashed), `stage-142310/` (the installed stage: edited drafts, `build/`, EDIT-*.json, STYLE-DIFF-CHECK.md, PROV-*, SCAN-*, VOCAB-DELTA-*, PH-*, INDEX-*, PDF text), `EDIT-TABLE.md` (the full before→after table), `SHA256-installed.txt`. Earlier stages (`dry-*`, `stage-142148`, `stage-142215`) are the iterations recorded in LOG.md.

## Status per item

| item | status | note |
|---|---|---|
| 7 — seed-5 limitation sentence | **partly** | Written in Sec. 4 (CMAZE-T, six seeds) with the mechanism the plan rules: not a hold-out effect; one outcome code (legal, move south, pocket 2); content-capacity failure of the 6-atom design; the 10-atom relational design reaches both tuples. **The 0.04 margin and "106 of the 317 fit rows" are NOT printed: neither number exists in any record on batmac** (searched `~/clu/evidence`, `~/clu/outputs`, `~/clu/stage`, `~/clu/tasks`, and every branch in git history; the only hit is PLAN.md itself). The "content-capacity" mechanism is therefore printed on the plan's authority alone; a `%` comment says so beside it. **For the Head:** supply the W23 diagnosis record to print the two numbers, or keep the sentence without them. |
| 7 — relational coding's limit, one sentence | **done** | n-CLU-REL paragraph, Sec. 4: "relations reach the n-CLU, colour identity does not … the value row stays a loss: n-CLU-REL reads 0.554 against the best matched baseline's 0.797 (six-seed means) and is behind on all six seeds." |
| 7 — "every number verified / traced" sentences scoped | **done** | Reproducibility statement and App. A "Provenance" paragraph, both scoped (1,308 of 2,293; presence, not meaning). The App. A sentence "Numbers are quoted as the record prints them; none is re-derived." is left as it was (it is about quoting, not checking). |
| 7 — two `% SRC` fixes (0.6407; "16 of 16") | **done** | Neither number is in the rendered text any more: both left with pw1-draft-figures (item 1: the fig:maze-wm caption) and pw1-draft-cuts (item 15: Fig. maze-rel-tuples). The fixes go on their removal pointers, and on the two surviving sentences that carry the seed-5 tie (details below). |
| 7 — PoE 27.135× | **unchanged, verified** | 2 occurrences per draft, `27.135$\times$` (Tab. lace-poe note; App. B), untouched. |
| 8 — no seed hard/easy | **done** | The one residual was "its seeds differ sharply in difficulty" (CMAZE-T lead-in), now "its results differ widely from seed to seed". pw1-draft-figures had already removed the other rendered hard/easy text, the hard-seed split and the ceiling paragraph. The CMAZE-T mechanism paragraph ("Why a hold-out draw can remove a whole colour relation") already names no seed and prints no ceiling number. Unchanged. |
| 13 — "exact" retired in the deletion sense; plain phrasing; scoping sentence; no general post-deletion-composition claim; limitations; future work | **done** | 47 operations per draft (title included). Rendered-source count of `exact|exactly|exactness`: **88 → 34** in each draft; every survivor is a non-deletion sense (list below). |

## Choices taken (open; logged)
1. **Title (FLAG FOR THE HEAD):** `naive-CLU: Exactly Deletable Compositional Generation\\ on a Frozen Encoder` → `naive-CLU: Deletable Compositional Generation\\ on a Frozen Encoder`. This is the minimal change. The Head may prefer a different title.
2. **Scoping sentence** (object section, the quote that states the definition): the small-caps lead reads "The match is on the n-CLU and its address book; the frozen encoder and decoder are outside the object." This is the plan's sentence. It is followed by the plain phrasing "the n-CLU matches an n-CLU that was never fitted on the deleted rows, and the address book likewise".
3. **No general post-deletion-composition claim.** I removed the abstract clause "Value and item deletions leave composition on the surviving combinations within one tuple per held-out combination" and the contribution-2 clause "Value and item deletion move composition … by at most 0.61". Contribution 2 now says "We make no general claim about composition after deletion". The measured survivor moves (−0.06 / +0.33 / −0.61) stay in Sec. 3.3 and Tab. del. The 0.61 left the contribution list with a pointer comment.
4. **Limitations printed:** the conjunction loss (limitation 2: 6.13 / 6.43 / 6.30 per held-out combination on the red sphere) and the address book's refit (limitation 3, retitled "Deletion is not cheap in memory or time: the address book's refit"; 0.97 GB, 2,367 to 5,913 s). Limitation 4 is retitled "Deletability requires a large number of bytes".
5. **Future-work sentence** is a new paragraph "Deletion." in Sec. 6 (Follow-up), after the physics-CLU paragraph. It says the n-CLU composes above every baseline retrained without the same rows on the value and item scopes, and level with the MLP and the LSTM on the red sphere, where its 95% interval includes zero on some seeds. It carries no new number except "95". Verified against `w20-shapes-data-matched/SHAPES-MATCHED.md` l.222–260 (the "ours − rival retrain-without | CI95" rows): value and item, all 24 lower bounds > 0; red sphere, the CI crosses 0 for mlp seed 1 [−0.94, 6.02], mlp seed 2 [−0.50, 5.67] and lstm seed 2 [−0.87, 5.82]. Every other red-sphere lower bound is > 0.
6. **Literature sentences** (no claim of ours, but "exact" in the deletion sense): "Exact unlearning is also established" → "Unlearning that matches a retrain without the deleted rows is also established". The screen's "none exact" → "none deleting training rows" (NOVELTY.md l.244: "None is exact, and none deletes training rows"). The orthogonal query `"compositional" + "generative" + "exact deletion"` (NOVELTY.md l.252) is now described as "compositional ∧ generative ∧ a deletion phrase". The query string itself is no longer quoted. That is a small loss of precision, taken to retire the word.
7. **"red $\wedge$ sphere" → "red sphere"** only in the abstract sentence I rewrote (the standing ruling). Other occurrences are outside my items and were left alone. "red $\wedge$ cube" is a different object.
8. **"exact" in the physics appendix's designed-memory paragraph** ("Memory-level exact deletion … deletion is exact at every load") is deletion-sense and was retired. The new wording is "leaves a memory byte-equal to one never written with the deleted items", which is what the paragraph's "byte equality 1.0000" states.
9. **App. E smoke:** "exactness \sci{…}" (3×) → "deviation from the refit \sci{…}". The values are unchanged.
10. Page count 41 → 42 in both drafts. Main text: Sec. 5 moves p. 16 → p. 17; Sec. 6 stays on p. 18. Length is MF-11, held by the Head.

## Item 7: every new number → file → path (verbatim)
| value printed | file (evidence root `~/clu/evidence/outputs/` unless noted) | path / line |
|---|---|---|
| 0.554 | `w21-rel-newseeds/TABLES.md` | §2 "The `+rel` DIAGNOSTIC", `#### row value`, l.134 `\| CLU \| 0.489 \| 0.501 \| 0.512 \| 0.525 \| 0.692 \| 0.604 \| 0.554 \|` (mean (6) column; the CLU of this section is CLU-REL-A10) |
| 0.797 | `w21-rel-newseeds/TABLES.md` | same block, l.147 `**best matched rival (frozen input, 8k/80k)** … \| 0.797 \|` (mean column) |
| "behind on all six seeds" | `w21-rel-newseeds/TABLES.md` | same block, row "CLU − best matched rival": −0.193 / −0.360 / −0.327 / −0.297 / −0.056 / −0.227 |
| 1,308; 2,293 | `~/clu/outputs/w23-fold/FINDINGS.md` (cited as `../../outputs/w23-fold/FINDINGS.md`) | l.46 "Live draft-cell coverage rose from **212 to 1,308 of 2,293** (`DRAFT-v2.tex`)" |
| "three groups of runs" | `~/clu/outputs/w23-fold/FINDINGS.md` | l.23–25 (bar-d-score / coverage-control / lace-poe-rows, "closes the three w15 spokes only") |
| (legal, move south, pocket 2) | `w21-maze-t-seed4/replay/check_BOTH_test_seed5_PERTUPLE.json` + `chlu/exp_w15_cmaze_t.py` | `PER_TUPLE_NWAY.CLU["2,0,1,2"] = [90, 0]`, `["2,0,3,2"] = [80, 0]` (tuple = pocket, door, tint, direction); `ACTIONS = "NESWP"`, so direction 2 = S; `outcome_of` gives (1, 1+2, 0, pocket) for a legal move with no key event; with no door, pocket 2 and tint 1 or 3, the move is legal (rule B fires only when pocket = tint) |
| "the 10-atom relational design scores every row of both tuples valid" | `w21-rel-newseeds/replay/BOTH_test_seed5_PERTUPLE.json`; `w21-rel-newseeds/config/CLU-REL-A10.json` | `PER_TUPLE_NWAY.CLU["2,0,1,2"] = [90, 90]`, `["2,0,3,2"] = [80, 80]`; `all_reproduce: true`; `n_atoms: 10` |
| "not an effect of the hold-out draw" | `w21-maze-t-seed4/FINDINGS.md` | l.122–125 "fully supported in train (class A) … This is a CLU-specific miss on reachable data, not a hard-seed effect." |
| 95 (future-work sentence, item 13) | `w20-shapes-data-matched/SHAPES-MATCHED.md` | l.84 / l.222 column header `CI95` |
| **0.04; 106; 317** | **NOT FOUND** | no record on batmac. **Not printed.** |

## Item 7: the two SRC fixes
| token | where it was | the gap (w23-fold) | fix |
|---|---|---|---|
| `0.6407` | fig:maze-wm caption (removed by pw1-draft-figures, item 1) | the line cited `MAZE-T-SEED4.json` (leaf `0.6407297594090047`); the rounded value is printed only in the uncited `w21-maze-t-seed4/TABLES.md` l.11 | The removal pointer already listed `w21-maze-t-seed4/TABLES.md`. Added a line naming l.11 (`cut = 0.6407`) and the JSON leaf `w21-maze-t-seed4/MAZE-T-SEED4.json` → `classification.cut.cut` = 0.6407297594090047 |
| `16 of 16` | Fig. maze-rel-tuples caption (removed by pw1-draft-cuts, item 15) | the cited `w21-rel-newseeds/FINDINGS.md` l.62 holds only the control half; the relational-baseline half is only in `w22-figures-redraw2/FINDINGS.md` l.15 ("also equals it on 16/16") | Added `../../outputs/w22-figures-redraw2/FINDINGS.md` to that removal pointer. Also added it to the SRCBLOCKs of the two surviving sentences with the seed-5 tie ("on seed 5 a matched baseline ties it", Sec. 4; "seed 5: a matched transformer at 80,000 steps equals the n-CLU", Tab. cmazet-sym): l.16 "`rival@10x-steps: transformer` = CLU = 0.9393939…, on 16/16 tuples". The seed-4 control tie ("equal on every held-out tuple") stays on `w21-rel-newseeds/FINDINGS.md` l.61–62, which carries it. |

## Before → after (every sentence changed)
The full table, with before and after text for each of the 56 operations, is in **`EDIT-TABLE.md`**. It gives lines in both drafts, whether the two drafts' texts are equal, and the numbers leaving or added. Counts per item, identical in both drafts: **item 13: 47; item 7: 8; item 8: 1; total 56 per draft.** 42 operations have the same before- and after-text in both drafts. 7 have a different before-text but the same after-text. At the other 7 sites, the style twin's own paraphrase is kept in the rewritten text. Every number-bearing new text is identical in both drafts (style diff-check on numbers EMPTY).

Short index (DRAFT-v2 line → site): 75 title · 91 abstract "forget … exactly", "exact unlearning" · 117 abstract deletion sentences · 162 positioning · 176, 180 novelty screen · 188, 190 SISA · 205 decremental PCA · 218–223 Sec. 1.2 definition (3 ops) · 239 contribution 2 · 323–337 object section (4 ops) · 719 Sec. 3.3 title · 793 retained statistic · 812 the probe · 831, 841, 844 Sec. 3.4 · 855, 901, 904, 905 Sec. 3.5 · 912 Sec. 3.6 · 989 CMAZE-T lead-in (item 8) · 1002 seed-5 (item 7) · 1018, 1032 n-CLU-REL SRCBLOCK and limit (item 7) · 1043 CMAZE-T deletion · 1093, 1098, 1101 limitations 2–4 · 1178, 1183 leaving note · 1195 Sec. 6 future work · 1207 ethics · 1213 reproducibility (item 7) · 1237 App. A provenance (item 7) · 1531, 1552, 1569, 1582, 1587–1590 App. E (6 ops) · 1830 pointer "16 of 16" (item 7) · 1964 App. physics · 2074 pointer 0.6407 (item 7) · 2150 Tab. cmazet · 2258 SRCBLOCK Tab. cmazet-sym (item 7) · 2264 Tab. cmazet-sym deletion row.

## Residual scans (rendered text: `pdftext.py` over the built PDFs, figure text included; same result in both drafts)
| token | before | after | residuals |
|---|---|---|---|
| exact / exactly / exactness (source, non-comment) | 88 | **34** | all non-deletion (list below) |
| exact* (PDF text) | — | 36 | the 34 above + **2 bibliography titles** of cited works: Quan et al., "Exact federated continual unlearning for ridge heads …" and Xu et al., "Netflix and Forget: Efficient and Exact Machine Unlearning …". These are other authors' titles. They cannot be reworded without misciting, so the bib was not touched. |
| hard seed / easy seed | 0 / 0 (already, after pw1-draft-figures) | 0 / 0 | — |
| hard | 1 | 1 | "Hard read isolation (designed memory only)": physics appendix, a memory property, not a seed |
| easy | 0 | 0 | — |
| difficult (seed difficulty) | 1 | 0 | — |
| ceiling | 16 | 16 | none is a per-seed / best-of-arms maze ceiling: Shapes3D count ceilings 36 / 72 (object section, Tab. t7 caption, App. F table ×2, encoder-loop caption); "validation sits near its 24-tuple ceiling"; "because it is a ceiling and not an arm" (item memory); nearest-prototype ceiling ×4 (Sec. 3.7, Tab. lace-poe); "the ceiling ladder" (App. F title); "oracle read (ceiling)", "absolute content ceiling at the pin", "Two ceiling rungs" (App. F); **Tab. cmazet (w16 asymmetric row): "Ceiling set by the request design, not the n-CLU: validation n-way 0.7247 … across about 15 configurations"**. That last one is a design plateau, not a seed label. The Head may want it reworded in PW-2. |

Surviving "exact" in non-deletion senses (DRAFT-v2 line; the style twin has the same 34):
- numeric equality: "collapsing to exactly 0.0000" (105), "read exactly 0.000" (413), "exactly constant" (572, 654), "read exactly 0.0000" (574), "exactly 16,384 B" (1590, byte residual), "exactly 0.0" (1874, 1901), "ties … exactly on the two symbolic refusal rows" (1972)
- "the same rows": "the n-CLU's exact 131,072-row pool" (416), "exactly the n-CLU's content rows" (588, 629, 669)
- exact-match requests: 954, 1129, 1338, 1382, 1776
- mathematics and physics: "exactly the zero-dynamics limit" (132), "$z_{req}$ exactly at λ=0" (1155), 1870, 1895, 1898, "exact inversion" (1926), "only the exact root" (1948), "bit-exactly" (1950, replication), "exact one-step Jacobian" (2000), "exactly zero off the coupling graph" (2011), "exact only at zero damping" (2054, reversible training), "exact value fidelity" (2059, trilemma), "exactly optimal" (2076), "exact-key" (2079, stitching), "exact quadratic" (2377), "were then exact" (2403, predictions)

Banned-word scan of the rendered captions: 0 hits in both drafts. Whole-draft vocabulary delta (scan.py, both drafts identical): exact 88 → 34; cost 8 → 7 (contribution 2's "with its cost and its limitation printed" is gone); Sec.~ 36 → 37 (the future-work paragraph's `Sec.~\ref{sec:del-comparators}`). No other token moved.

## Both drafts (stage-142310, built from the w17-paper layout)
| check | DRAFT-v2 before → after | DRAFT-v2-style before → after |
|---|---|---|
| exits (pdflatex, bibtex, pdflatex ×3) | 0 → 0 | 0 → 0 |
| undefined citations / references | 0 / 0 → 0 / 0 | 0 / 0 → 0 / 0 |
| errors / overfull boxes | 0 / 0 → 0 / 0 | 0 / 0 → 0 / 0 |
| bibliography items | 55 → 55 | 55 → 55 |
| pages | 41 → **42** | 41 → **42** |
| "Figure pending" | 0 → 0 | 0 → 0 |
| provenance (unsourced lines / not-found tokens) | 7 / 8 → 7 / 8, **identical lists**; checked tokens 2,590 → 2,594 | 7 / 9 → 7 / 9, **identical lists**; 2,590 → 2,594 |
| `\PLACEHOLDER` / `\TODO` | 2 / 4 → 2 / 4, **byte-identical, same order** | 2 / 4 → 2 / 4, byte-identical |

Style diff-check (stage-142310/STYLE-DIFF-CHECK.md): numbers EMPTY DIFF; placeholders EMPTY (ordered byte-identical True); TODOs byte-identical True; artefact paths EMPTY; provenance comment strings EMPTY and in identical order (437 vs 437, was 435); headings identical (94, was 93: the new "Deletion." paragraph); labels identical (65); tables/figslots byte-identical (31); preamble identical.

The abstract's maze `\TODO` is byte-identical: all four `\TODO`s compare equal to the snapshot. The abstract's other changes are the item-13 deletion sentences only.

`\PLACEHOLDER` / `\TODO` after (DRAFT-v2; unchanged): TODO abstract CMAZE-T sentence (Head) · TODO byte ratio of the composition-only n-CLU · TODO AI use statement · TODO anonymised code archive · PLACEHOLDER observation-only per-model $s_{t+1}$ (families) · PLACEHOLDER observation-only per-model $s_{t+1}$ (door-then-tint). Full text: stage-142310/PH-*.txt.

Section start pages after (DRAFT-v2): 1 p.2 · 1.2 p.3 · 2 p.4 · 3 p.6 · 3.2 p.9 · 3.3 p.12 · 3.4 p.13 · 3.6 p.14 · 4 p.15 · 5 p.17 (was 16) · 5.3 p.18 · 6 p.18 · App. A p.24 (was 23) … App. S p.41. Style twin: 3.3 on p.11, otherwise the same. Figures: 1 p.4, 2 p.5, 3 p.8, 4 p.9, 5 p.11, 6 p.27, 7 p.29, 8 p.31, 9 p.36, 10 p.38, 11 p.39, 12 p.40, 13 p.41 (stage-142310/INDEX-*.txt).

## Installed
The lock guard was clean before the write: live sessions `authtest`, `hubhost`, `pw1-draft-wording`, `pw1-visuals-tooling`, and none matches the lock patterns. Before the install, the live drafts and PDFs were cmp-equal to `snap-before/`. The stage files were copied, then cmp-verified at the destination (4/4 equal).

| file | before | after |
|---|---|---|
| DRAFT-v2.tex | `6d89e6f6683ce249e948ab074413dded93b3fca2e7d507f622b994cc279bf81c` | `2f720bae5a950f52816ee15871779980ecf882df661b3cf756370ff55d0ea568` |
| DRAFT-v2-style.tex | `c86387cf5b39823f0de6ccfdc304f7322ce14695951d27101ca450fa5c9e6918` | `3fa390b551853148699c05de3995974f1dd733d7dad2343440a8c0318cba0ca6` |
| DRAFT-v2.pdf | `5269a291ff3fc646b1057f9404cf16f686b9cb03b279ddd31b896d692b547f58` | `588d02f7c9708fbfbcdb6090499bd5ee27ef676b1fb618d5f0eaa17aef940069` |
| DRAFT-v2-style.pdf | `339e3e778a9a665da7d4039883c916cd68ffca8b499b01a1bf9bd50263b8416d` | `f9cf0ff15d96f008387689961851be2e8e94d1e9a2e665437f01dc6207b5a471` |

## For the Head
- **Title** changed (choice 1). Please confirm or reword.
- **Seed-5 numbers:** the plan's 0.04 margin and 106 of 317 are not in any record on batmac, so the sentence prints the mechanism without them.
- The "content-capacity" wording rests on the plan text alone. On batmac I verified: the outcome code, the 10-atom design scoring both tuples in full, and "not a hold-out effect".
- Bibliography titles containing "Exact" (2) remain; they are other authors' titles.
- Tab. cmazet's "Ceiling set by the request design" (w16 asymmetric row) remains (a design plateau, not a seed label).

## What did NOT run
- **The 0.04 margin and the 106 / 317 count were not printed**: no record on this machine carries them. No measurement was commissioned (zero runs), per COMMON rule 7.
- No figure was touched: `figures/` and `paper-visuals/` are pw1-visuals-tooling's. So the figure images were not scanned for deletion-sense "exact" beyond the PDF text extraction, which found none in figure text.
- `references.bib` was not edited (the two cited titles).
- Stale `.aux/.bbl/.log/.out` in `w17-paper/` were not rebuilt in place (only the two .tex and two PDFs were installed, as in the previous draft passes).
- Nothing held by the Head was applied: the abstract's maze `\TODO` (byte-identical), M3's deletion wording beyond item 13, MF-11 (length; the draft grew 41 → 42 pages), and MF-12.
- No palette or figure build (no plot touched). No measurement, fit, GPU or model load. The registry, claims matrix, TABLES.md and every `outputs/w*/` directory were only read.
- pdftotext is absent; the PDF text was extracted via `uv run --with pypdf` (pw1_draft_cuts/pdftext.py).

## Branch and bundle
Branch `pw1/draft-wording` at `29be0f8` (tooling only: `pw1_draft_wording/`; no outputs committed). The committed `edit.py`, re-run on `snap-before/`, reproduces both installed drafts byte-for-byte (`repro-*/`). Bundle `pw1-draft-wording.bundle` sha256 `a029186af212a5cee7913aa562f6cc66e2858529eb6da63c90d0c7c1fb5a1405`; `git bundle verify`: okay, complete history.
