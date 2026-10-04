# w21 `spoke/figures-redraw` — BRIEF (batmac)
**Wave-plan §6a. Base = `w19/maze-t-symmetric` tip 0455961. ZERO scored runs. ⛔ Never write to the ledger.**

## THE ONE JOB
**Every figure in the paper is drawn from artefacts, carries no banner, rebuilds byte-identically, and its caption and the placeholder index say what the figure actually shows.**

## PART A — the three non-maze figures (pure redraw, no model loading)
1. **crossover** — **mean ± seed sd at EVERY K**, from the 3-seed table. ⛔ No single-seed line anywhere.
2. **deletion** — **3 seeds**.
3. **ladder** — with the **data-matched MLP row** and the rival column sets **as they land**. ⭐ Build the script so a new rival set is a data file + a caption line, ⛔ not a code rewrite: `spoke/ladder-rivals-32k` will add a **third** set (32,768 rows) beside the 32,256-row and 131,072-row sets, and a later pass must be able to drop it in. **Pool sizes go in the caption** for every set.

## PART B — the maze results figure becomes the CONDITION TABLE
Replace the w16-only panels with one table figure, in this row order:
| row | label |
|---|---|
| w16 pixel-rivals | ⛔ **labelled ASYMMETRIC in the row itself** — rivals read a 636-dim observation, the CLU read the 318-wide design |
| words-main | LOSS |
| words-full | **LOSS** (w21 TEST) |
| observation | LOSS |
| words-plus-observation | (w19 A1) |
| words-plus-observation, improved | ⭐ **added only when `w21-maze-t-clu-improve` lands** (a second pass) |
⭐ **Per-seed markers on every row.** ⭐ Labels by **the Head's rule**: WIN (ahead on the mean and a majority of seeds, store gate holds) · COMPARABLE (within one seed's spread) · LOSS (behind). ⛔ **No numeric margin bar anywhere.**
⚠ Sources, and use the ARTEFACT not a report's prose: `evidence/outputs/w19-maze-t-symmetric/` (`val/BARS-*.json`, `test/BARS-BOTH-test.json`, `TABLES.md`) and `evidence/outputs/w21-words-full-test/` (`test/BARS-WORDS-FULL-test.json`, `TABLES.md`). w16's row: `evidence/outputs/w16-maze-wm/`.

## PART C — per-arm panels from the PERSISTED objects
⛔ The engine-only **hatched columns must go.** Draw the **CMAZE-T per-arm success/failure panels** and the **post-deletion panel** from wave 19's persisted **words-plus-observation (BOTH)** objects: CLU, every rival, every control, the blind arms, and the deletion pair.
**Staged for you at `~/clu/stage/w19-persist/BOTH_seed{0,1,2}/test/`** — 270 files, 3 seeds.
- ⭐⭐ **ASSERT THE HASHES BEFORE DRAWING.** Each object has a `.sha256` carrying **two** lines: `param_sha256 <hash>` and `file_sha256 <hash>  <name>`. Check **file_sha256** against the file, and check the object against `MANIFEST.json` (which also carries `bytes` and the `rebuild` args). ⛔ A single mismatch = STOP and report; do not draw from it.
- ⭐ The w19 BOTH CLU is **byte-identical to the scored wave-16 object** — say so in the caption.

## ⭐⭐ THE CHECK THAT CAN STOP PART C — do it first
The JSONs carry **no per-item results**, so per-arm panels require loading each object and re-reading. That is a different machine from the one that scored it.
**Before drawing anything, reproduce w19's BOTH TEST n-way from the loaded objects: `0.800 / 0.566 / 0.808` (mean 0.7246), controls `0.000 / 0.225 / 0.244 / 0.166`.**
⛔ **If it does not reproduce to the printed digits, STOP Part C, report the discrepancy, and still deliver Parts A, B and D.** ⛔ Do not draw a panel from a read you cannot reproduce. ⭐ Parts A/B/D do not depend on this.

## PART D — banners, captions, index
- ⛔ **Remove every in-image banner** (three figures carry `PLACEHOLDER · w≤14 evidence · wave 18 regenerates`) and **all pipeline language** from rendered text.
- Correct **both drafts' captions** and the **placeholder index**. ⚠ The draft header comment still says "Eleven PLACEHOLDERs remain" — it is a stale comment; the in-text index is correct.
- Update `figures/PROVENANCE.json` and `figures/FIGURES.md`.

## ACCEPTANCE
- **Two independent builds are byte-identical** (SHA256 manifest, as `figures/build.sh` already does). ⛔ Exit 0 proves nothing.
- Every figure traces to a named artefact; every number in a caption is in that artefact.
- Both drafts still compile: **exit 0, 0 undefined citations, 0 undefined references**, and **all four style diffs EMPTY**.
- ⛔ **No `\PLACEHOLDER{}` argument altered** and no paper prose rewritten beyond captions and the index.

## ⛔ THE `.tex` LOCK
Both drafts are single-writer. **Refuse to start if any peer session is live**, and check by name:
```bash
for g in w18-seeds-and-tables w18-placeholder-fill w20-citation-prune w20-referee3-fixes w20-referee4-fixes w21-placeholder-fill; do
  tmux has-session -t $g 2>/dev/null && { echo "ABORT: $g is live"; exit 1; }
done
```
⭐ Test every edit on a scratch copy first, verify, then install byte-identically.

## OUTPUT
`~/clu/outputs/w21-figures-redraw/` — plus the figures themselves into `~/clu/outputs/w17-paper/figures/`.
