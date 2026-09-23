# FIGURE-SLOTS.md: w19 spoke `draft-figure-slots`

**Verdict: DONE.** Both drafts were updated by **pure insertion**: every original line is kept, unchanged and in order, and this is asserted against the pre-edit backup. Both drafts compile with bibtex, with 0 errors, 0 undefined references or citations, 0 overfull boxes and labels settled. The logs are saved. All four style diffs are **EMPTY**.

| file | before (sha256) | after (sha256) | lines |
|---|---|---|---|
| `w17-paper/DRAFT-v2.tex` | `3ddf56a9…` | `33d1b530…` | 2,539 → 2,657 |
| `w17-paper/DRAFT-v2-style.tex` | `dc3a2bca…` | `9864922f…` | 2,528 → 2,646 |

Pre-edit copies: `backup-165314/`. The insertion is done by `insert_slots.py`, which uses label anchors and writes identical bytes into both files. There was one follow-up edit, a same-line `% SRC:` on the w19 table caption. `insert_slots.py` carries it.

## Figures inserted (identical in both drafts; figure numbers and pages are from the final compile)

| # | label | file | location | page v2 / style | source artefact (named in the LaTeX comment above the slot) |
|---|---|---|---|---|---|
| Fig. 1 | `fig:task-shapes` | `figures/fig_task_shapes.pdf` | top of §2 *The object, the judge and the ledger*, right after `\label{sec:object}`. This is **before §3, the first results section (p. 6)** | 4 / 4 | `w19-task-figures/TASK-FIGURES.md`, `proof/PROVENANCE-task-figures.json`. The figure contains no result |
| Fig. 2 | `fig:task-cmazet` | `figures/fig_task_cmazet.pdf` | same place, after Fig. 1 | 5 / 5 | same. CMAZE-T engine at `9f80f59`. The figure contains no result |
| Fig. 3 | `fig:t7-per-arm` | `figures/rendered/rc_w19_t7_per_arm_success_failure.pdf` | directly after Table 1 (`tab:t7`, §3.1) | 7 / 8 (table: 6 / 7) | `figures/rendered/{REQUESTS-w19,SAME-DECODER-w19}.json`, `README-w19.md`, `w19-success-failure-panels/REPORT.md` |
| Fig. 4 | `fig:t7-rival-wins` | `figures/rendered/rc_w19_t7_rival_succeeds_clu_fails.pdf` | after Fig. 3 | 8 / 9 | same (the `t7.counts_per_seed` block) |
| Fig. 5 | `fig:t7-rival-wins-s1` | `figures/rendered/rc_w19_t7_rival_succeeds_clu_fails_seed1.pdf` | after Fig. 4 | 8 / 9 | same |
| Fig. 12 | `fig:cmazet-families` | `figures/rendered/rc_w19_cmazet_families.pdf` | App. *Maze world-model rows*, directly after Table 26 (`tab:cmazet`) | 40 / 40 | `REQUESTS-w19.json`, `README-w19.md`, `REPORT.md`, `NEEDS-ARTEFACTS.json` |
| Fig. 13 | `fig:cmazet-door-tint` | `figures/rendered/rc_w19_cmazet_door_then_tint.pdf` | after Fig. 12 | 40 / 40 | same |
| Fig. 14 | `fig:cmazet-deletion` | `figures/rendered/rc_w19_cmazet_deletion.pdf` | after Fig. 13 | 41 / 41 | same |
| Table 27 | `tab:cmazet-sym` | (placeholder table) | after Fig. 14, before Table 28 (`tab:cmazeo`) | 41 / 41 | `tasks/w19/maze-t-symmetric.md` (design only) |

All `\figslot` calls are captured by the style checker's float test. The checker reports them byte-identical and in the same order in both files (42 vs 42 floats).

⭐ **The finding that must not be lost is printed** in the caption of Fig. 4, with its per-seed breakdown. On all 5,688 held-out requests per seed, a learned rival renders validly where the store does not on **30 / 24 / 30** requests at seeds 0 / 1 / 2: MLP 26/11/3, LSTM 13/13/25, transformer 2/3/6, continuous MLP 0/1/0. The store is valid on 5,177 / 5,167 / 5,354. The lookup floor does so on 67/26/46. Fig. 3 also shows the store's own failure row. All these numbers are w≤14 evidence (the re-render of the w12/w12b T7 legs). Each carries a `% SRCBLOCK` to `REQUESTS-w19.json`, and the provenance checker finds every one there.

## Every w19 / not-buildable placeholder inserted (10). Each names its row and says it may land as a negative or not at all

1–3. **CMAZE-T arm columns** (inside the captions of Figs. 12, 13 and 14). The per-arm generated s_{t+1} and the post-deletion per-arm outputs are *not buildable from persisted artefacts*, so they may remain undrawn, and if drawn may show the store failing. The figures are **engine-only**: s_t + action + true s_{t+1}, with the arm columns hatched. The captions say that no arm's output is drawn.
4–10. **Table 27, w19 `maze-t-symmetric`**, one row each: WORDS-MAIN, WORDS-FULL, OBS (with extractor accuracy and the extractor-appended rival row), BOTH (val; scored only if runs remain), rival budget + 10× diagnostic, deletion on the winning condition, and rollout validity printed once. The table has its own float and caption, and states that it contains no row of Table 26 (w16).

**Untouched:** the four `\PLACEHOLDER{}` owned by `bar-d-score` (×2) and `lace-poe-rows` (×2). Their arguments are byte-identical to the backup in both files (asserted).

## Style diff-check (`~/clu/outputs/w17-style-paraphrase/style_diff_check.py`, full output: `STYLE-DIFF-CHECK-w19.md`, exit 0)
1. numbers: **EMPTY DIFF**
2. placeholders: **EMPTY DIFF** (14 = 14, ordered and byte-identical)
3. artefact comments: **EMPTY DIFF** (paths and whole comment strings; 349 = 349 in the same order)
4. section order: **EMPTY** (105 headings identical; 80 `\label`s identical)
Extras: floats byte-identical, preamble byte-identical. **ALL DIFFS EMPTY.**

## Compile (performed here; logs saved)
`pdflatex` → `bibtex` (with `BIBINPUTS/BSTINPUTS` = `w17-paper/`) → `pdflatex` ×3, from `w17-paper/`, output to `compile-final-2/`. Saved: `compile-DRAFT-v2{,-style}.log`, `.blg`, `.pdf`. Both are 44 pages with 0 errors, 0 undefined, 0 overfull and no "labels may have changed". Baseline, before any edit: `compile-baseline/` (both compiled, exit 0).
⚠ **Pre-existing, not introduced by me:** the w17 `compile-v2*.log` files have 226 undefined citations. bibtex had never succeeded, because `\bibliographystyle{iclr2027/iclr2027_conference}` is not found without `BSTINPUTS`. With it set, every citation resolves.

## Provenance checker (`check_provenance.py`, before vs after, `prov-{before,after}-*.txt`)
The findings are identical before and after insertion (6 unsourced lines, 1 not-found token, all pre-existing). My inserted numbers are all sourced. w19 artefacts are not staged in `~/clu/evidence/outputs/`, so their SRC paths are written relative to EV as `../../outputs/...` and `../../tasks/...`. They resolve, and the checker finds them.

## Flagged, not acted on (prose, which this pass may not change)
- **Placeholder index (appendix `app:placeholders`)** still says "Four rows remain pending", and the file header comment says "Four PLACEHOLDERs remain". There are now 14 (4 + 10 new). `placeholder-fill` pass 2 or the Hub should update that prose.
- **AMENDMENT-1 vs the draft:** the draft already carries w15/w16 numbers (written in by w18 placeholder-fill, per its header). I added no w15/w16 number. The CMAZE-T engine figures carry no w16 score.
- `rc_w19_cmazet_deletion` has a cosmetic defect in the figure itself: its "retrained without" column headers overlap the group labels. That belongs to `success-failure-panels` to fix; I did not edit it.
- w17's rendered-compositions sets (`rc_set1/2/3`, `rc_deletion_red_sphere`) are **still not slotted** in either draft. That was outside this pass's brief (w19 panels only).

## DID NOT RUN / NOT DONE
- No CMAZE-T arm output is drawn anywhere (declared not buildable upstream). No scored run and no compute.
- No figure file was modified. No prose word was changed. The Placeholder index was not updated (it is prose).
- No compile of any other file.
