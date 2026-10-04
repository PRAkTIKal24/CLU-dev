# WAVE 18 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-24. Five spokes, twelve passes, machine `batmac` (Apple M4). Base `base/iter8`.**
**ZERO scored runs. ZERO compute beyond `pdflatex`/`bibtex`. Zero GPU, zero cloud.** This wave changed no number; it audited, folded and edited.
Record: `.claude/outputs/w18-{referee-4,referee-fixes,placeholder-fill,registry-fold,seeds-and-tables}/`. Code landed from **each spoke's own bundle**: `w18/seeds-and-tables = f52e794`, `w18/registry-fold = 9fc536b`. ⚠ Each bundle carries **five** w18 branches at different times — ⛔ never fetch a spoke's branch from a peer's bundle.
⭐ The Hub re-derived the load-bearing numbers from the primary artefacts. One of them **falsified a claim in the Hub's own w19 report**, which has been corrected (§5).

---

## 0. The scoreboard

| spoke | passes | verdict |
|---|---|---|
| ⭐⭐ `referee-4` | 1 | **4/10 REJECT as submitted, confidence 4/5.** 12 must-fix, 14 should-fix |
| `referee-fixes` | 1 | M2, M4, M5, M6, M7(e) **applied** to both drafts; **M1 and M3 HELD** for the Head |
| `placeholder-fill` | 2 | the 4 w15 placeholders **filled, not deleted**; exactly **10 w19 placeholders** remain, byte-identical and in order |
| `registry-fold` | 4 | `N366`–`N371` added; **363** registry entries, **0 duplicates**; matrix **v2.25 PROPOSED** |
| `seeds-and-tables` | 4 | the Head's deletion-rule amendment applied to tables **and both live drafts**; **770 verified cells** |

---

## 1. ⭐⭐ THE HEADLINE: the referee predicted the CMAZE-T collapse, and wave 19 then measured it

**MF-1 — "the CMAZE-T store row gives the store a different input from the rivals, the abstract leads with it, and the asymmetry is not disclosed."**
This was written **before** w19 ran. w19 then gave every arm the identical input and the margin vanished. ⭐ **The referee's prediction and the measurement agree**, which is the strongest evidence in this wave that the review process is working.

⛔⛔ **And MF-1 carries a second charge the Hub had not seen stated anywhere else — a double standard against the paper's own tripwire:**
- On CMAZE-2, a BLIND-SEQUENTIAL arm at **0.9077** was taken as proof the request layer was broken, and the row was voided (§4.2).
- On CMAZE-T, the **composition-blind BLIND-SEQUENTIAL *planner* reads 0.9071 — above the store's 0.7246** on unseen n-way. The w15 harness shipped the warning by name (`composition_blind_PLANNER_scores_near_the_clu_on_unseen_nway`).
- §4.4 prints only the **learned** blind arms (0.3418 / 0.0716), not the planner.
⭐ **The paper's own CMAZE-2 standard voids rows for less.** This is the Advisor's to rule on, and it is sharper than anything in w19's own report.

⭐⭐ **M1 is now the THIRD instance of the same defect, on a new axis each time** (registry `N339`, `⟲ PASS 4`):
| instance | what the rival was given less of |
|---|---|
| Table 1, Shapes | the **data pool** (16,384 against the store's 131,072) |
| w16 CMAZE-O | the **training steps** |
| w16 CMAZE-T | the **input** — and this one **reverses the headline** |

## 2. ⭐ THE REFEREE'S 12 MUST-FIX — triaged

**Applied this wave (by `referee-fixes`, both drafts):** M2, M4, M5, M6, M7(e); M7(c) partly (SISA cited; the CDM entry handed on).

**Landed later in the wave:** MF-5's stale text and MF-2's pool statement were fixed by `seeds-and-tables` passes 3–4, along with the MF2 rounding slip (`23.5117 → 23.5121`) and the `tab:k5` inference note.

⛔ **HELD FOR THE HEAD, untouched by every spoke:** **MF-9 (M3** — "exact deletion" defined on an object whose address book is not refit**)**, **MF-11 (length)**, **MF-12 (venue items)**.

⭐⭐ **MF-2 was the wave's live risk, and it is CLOSED.** The referee's number-one immediate action was *"correct the w20 brief's ladder pool now, before w20 scores"* — the brief (`tasks/w20/shapes-data-matched.md` l.9) says the store's pool is 131,072 rows, but the **ladder** store is fitted on **32,768** (`n_store_train`). As written, w20 would have refit the ladder rivals on **4×** the store's data and **reversed** the asymmetry it exists to remove.
⭐ **The Hub verified w20's own PREREG and it already overrides the brief**: *"data-matched to **32,768** rows, NOT to 131,072, and every caption says so"*, with the M-16k third column moved 32,256 → 32,768. w20 measured the real pools and corrected itself. ⚠ The stale line remains in the task file and should be annotated once w20 finishes, so it is never read again.

**Still open and unruled (the Advisor's):** **MF-3** — the paper already holds a **data-matched, read-path-matched rival and never prints it**, leaving the headline without insurance. **MF-7** — LACE and PoE sit **inside the headline table while performing at control level**. **MF-8** — the abstract, contribution 1 and the leaving note **overstate the headline relative to the paper's own disclosures**. **MF-10** — K\* reads as a clean selection in the abstract while the main text is honest about the plateau.

## 3. The length and compile picture — measured, not estimated

| check | result |
|---|---|
| Compile, both drafts | **exit 0, 0 undefined citations, 0 undefined references**, 55 `\bibitem`s |
| Total pages | `DRAFT-v2` **45**, `DRAFT-v2-style` **46** (47 each after the pass-4 edits) |
| **Main text** | **≈ 18 pp** on both, measured with `\typeout{\thepage}` markers |
| Against ICLR 2027's 9 pp strict limit | ⛔ **over by ≈ 9 pp — twice the limit.** REFEREE-v2 measured 12 pp; the growth is the task figures, the per-arm panels and the w15/w16 fill-in |
| The two drafts agree | 54 identical headings in order, identical label order, 11/11 placeholders, 5/5 `\TODO`, identical citation keys, **identical multiset of result numbers** |

⚠ **Two build traps worth carrying:**
1. ⛔ `\bibliographystyle{iclr2027/iclr2027_conference}` **hard-codes the sub-folder**. A build from any other layout **silently produces an empty bibliography** — the referee's own first build did exactly that. **Build from `w17-paper/`.**
2. ⛔ **Anonymity: the PDF is clean** (no name, institution, URL, e-mail or path in any rendered line; `/Author()` and `/Title()` empty) — **but the `.tex` COMMENTS carry `_thinkpad/…` and `C:\chlu…` paths.** Harmless in the PDF; ⛔ **the source must never be uploaded as-is** to arXiv or as supplementary material.

## 4. What the registry fold established

- **Additive, and checked mechanically:** 299 lines inserted, **0 replaced, 0 deleted**; **363** `N` entries with **zero duplicate numbers**. Matrix: 112 inserted, only the title line replaced and the old title kept verbatim as its tail. Every decimal and every 4+-digit token in the new text (123 registry + 41 matrix) verified verbatim against source.
- ⭐ **w19's 384 persisted objects** are each hashed and manifested with **0 mismatches**, closing `N363`'s defect class for w19. ⚠ Three limits: only the 18 manifests are on batmac (the objects are on the ThinkPad); **w16's rivals are not recoverable** because the BOTH rivals read a different input; and the `persist_dir` pointers are stale pre-move paths.
- ⛔ **C-512 was not re-run and is exposed to the same asymmetry as A1** — it may **not** be promoted in A1's place.

## 5. ⚠⚠ `N371` — where a wave report disagreed with its own primary record, including this Hub's

The curator checked w19's prose against w19's JSONs and found **six** disagreements. Two matter:

1. ⛔⛔ **"On every condition the four charter-C controls collapse" is FALSE.** On **WORDS-MAIN** the control bar **fails**: worst control **0.18744** against a bar of **0.18437**, `B2_store_gate = false`. w19's own §2 table printed this correctly; its §0 prose contradicted its own table.
   ⚠ **The Hub repeated the prose in `w19-advisor-report.md` and has corrected it** (§2 and §7 of that report now read "three of the four conditions", with a dated `⟲ CORRECTION` note naming this source). ⭐ No other number in the w19 report changed.
2. ⚠ **VAL and TEST margins were conflated.** The VAL margin is **+0.1079 (a PASS)**; **+0.0988 is TEST**. The VAL knife-edge is the seed-1 lead (0.56889 against 0.56741). ⭐ The Hub's w19 report had this right.
Also corrected: "786 objects" is **384** (786 counts objects + `.sha256` + manifests); "634 / 29 dyn" is **636 / 31**; and "the extractor recovers the whole word tuple" drops the qualifier that it does so on only **6–22 %** of held-out n-way rows.

⭐⭐ **The lesson, and it is now in the operating docs: a paraphrase is not a source, and a wave report written from a summary disagrees with its own artefact.** This is the second time (`N351` was the first, eight disagreements).

## 6. What the fill actually says — two results that are NOT wins

- **BAR D: MIXED, no claim in either direction.** Growth is ~0 with CI95 on both rows, 0 VOID seeds, controls collapsing. ⭐ The reason it could not show is printed: `loop_swap` **saturates** (store 1.000/0.995/0.998 of ceiling, learner 1.000/0.991/0.997) — labelled **"descriptive, not a bar"**.
- ⛔ **The CPU leg is UNMEASURED, not void** — its `rival_mlp_180k` 0.000 rows are the **frozen-feature** rival (`N359`). Re-scoring is declared not-run.
- **Encoder decomposition: ORDERING NOT REPRODUCED** (A′ < 0 on 2/6, A″ > 0 on 6/6). The swap-loss half survives. ⛔ **"Unfreezing hurts" is not a general finding** — it is a property of one seed's well-trained `alone` control. The features improved for **every** arm, so this is a result about the encoder, not the store.
- ⚠ `referee-fixes` measured, against the strongest learned rival available: **ladder n=2 seed 2 is 33.633 against 24.250** — far smaller than Table 1's +39 to +49. ⛔ That stronger rival set was fitted on a pool **containing the held-out values**.

## 7. Integrity checks the spokes ran on themselves

- `placeholder-fill`: all four style diffs **EMPTY**; provenance findings **identical to the pre-pass baseline** (2,786 tokens); every inserted number found verbatim in the file its `% SRC` names; pure insertion — the only 32 removed lines are placeholder arguments and bookkeeping.
- `seeds-and-tables`: **770 cells verified**, two builds byte-identical, and every negative test (mutated cell, wrong draft line, bare number, orphan token, tampered historical cell) **exits 1**. Draft edits tested on scratch copies first, then applied byte-identically to the live files.
- `registry-fold`: difflib additivity proof, duplicate-free headings, mechanical number audit.
- Across the wave: **no sub-agents, no background jobs left running, no process killed, `tmux kill-server` never used.**

## 8. What did NOT run — declared

- ⛔ **No scored run, no compute, no GPU, no cloud.** Only `pdflatex`/`bibtex` and figure rebuilds.
- ⛔ **w20 `shapes-data-matched` is NOT folded into the registry** — it was still running. **A fifth `registry-fold` pass is required**, and nothing about w20 was guessed.
- ⛔ **The w15 `bar-d-score` / `coverage-control` / `lace-poe-rows` fold into `TABLES.md` was declared out of brief by passes 3 AND 4.** ⚠ **Those numbers are in the draft but not yet verified by `TABLES.md`** — a real verification gap. ⭐ The Hub has scoped it as **one** `seeds-and-tables` pass 5 together with the w20 cells, rather than two spokes.
- Not verified or rebuilt: any w19 persisted object (they are on the ThinkPad, not batmac). `logs/` is empty on batmac. `TABLES.md` and `MAZE-T-SYMMETRIC.json` were hashed, not audited cell by cell.
- Not done and named: MF14(i); the pass-2 caption patch on the live draft; the `tab:cmaze-del` label collision; `CORRECTIONS-ABSORBED.tex`'s archived `23.5117`; a visual PDF inspection (no rasteriser on that machine); the `N359` re-score; the C-512 re-run.
- ⚠ The draft header comment still reads "Eleven PLACEHOLDERs remain" — a **comment line, now stale**; the index in the text is correct.

## 9. What the Advisor is asked to rule on

1. ⭐⭐ **The blind-planner double standard (MF-1).** A composition-blind planner reads **0.9071** against the store's **0.7246** on CMAZE-T unseen n-way, and the harness shipped the warning by name. On CMAZE-2 a 0.9077 blind arm voided the row. Does the same standard now void the CMAZE-T row — and if not, what distinguishes them?
2. ⭐⭐ **MF-3: the paper holds a data-matched, read-path-matched rival and never prints it.** Printing it is the insurance the headline lacks. Is there a reason not to?
3. ⭐ **MF-7: LACE and PoE perform at control level inside the headline table.** The referee's minimal fix is to move them to their own appendix table, keep one honest sentence, and list "no working generative baseline" as a limitation. Accept?
4. ⚠ **MF-8 and MF-10: the abstract overstates relative to the paper's own disclosures**, and K\* reads there as a clean selection although the main text is honest about the plateau. Both are abstract-level wording.
5. ⚠ **C-512 may not stand in for A1** — it is exposed to the same input asymmetry and was never re-run. Confirm it is not promoted.

⛔ **Held for the Head and untouched:** M3 (MF-9), the 18 pp → 9 pp trim (MF-11), and the venue items including the AI-use statement (MF-12).
