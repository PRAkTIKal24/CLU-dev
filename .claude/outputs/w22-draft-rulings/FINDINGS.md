# w22 `spoke/draft-rulings` — FINDINGS (batmac, 2026-09-24)

**Zero compute and zero scored runs. Nothing was written to a ledger. No PREREG was written, and no bar or label was declared.**
The Head's rulings 1–12 and AMENDMENT-1 were applied to both drafts. The edits were staged (`staged-4/`), compiled, checked and then installed byte-identically at 20:42:10Z.
- Tools: `w22_draft_rulings/` on branch `w22/draft-rulings`. `apply_rulings.py` + `texts.py` make the edits; `run_checks.sh` and `retired_grep.py` do the checks.
- The replaced regions are listed in `staged-4/REPLACED-REGIONS.md`. The full diffs are `DRAFT-v2.diff` and `DRAFT-v2-style.diff`.

## Acceptance
| check | result |
|---|---|
| Compile, both drafts, built in place from `w17-paper/` | exit 0 · 0 undefined citations · 0 undefined references · 0 errors · **0 overfull** · 0 "Float too large" · 0 "Figure pending" · 55 bibitems · **48 / 49 pp** (was 51 / 52) |
| Style diffs (`STYLE-DIFF-CHECK-final.md`) | **All four EMPTY**: numbers, placeholders, artefact paths, section order. 109 = 109 headings; 79 = 79 labels; floats byte-identical 42 = 42 |
| Provenance checker | Findings are **identical to the pre-edit baseline** in both drafts, apart from line shifts. 3,033 number tokens were checked (baseline 2,828). Every new number is found in the file its `% SRC` names |
| Install | Before install, the live drafts were `cmp`-equal to `snapshot-20260924T201907Z/` and the lock was re-checked. After install, the drafts are `cmp`-equal to `staged-4/` (sha256 `3b031b0f…` v2, `8e491e18…` style) |
| Retired numbers in **rendered** text (`RETIRED-GREP-installed.txt`) | The PDF text is extracted with Ghostscript, and **whole** numeric tokens are compared with ~150 retired values (4 classes). **0 hits** in the margins, the 1x/own-pool values, "NegGrad" and "16,384" (only the unrelated "16,384 B" byte residual remains). Positive control on the pre-edit PDF: 21 + 43 + 51 + 29 hits, "NegGrad" ×4, "16,384" ×16. **5 remaining exact-token coincidences per draft, all unrelated measured values:** `37.550` (w20 data-matched continuous MLP, n=5, seed 2), `0.6667` ×2 (CMAZE-2b MERGE, seed 1), `0.1020` (w16 CMAZE-T rival retrain), `164` (a review-mode line number) |

## Placeholder and TODO counts (rendered)
- **PLACEHOLDER: 15 → 4.**
  - **Filled (11):** the w20 headline (AMENDMENT item 1); w20 deletion (b) on value, red sphere and item, and (c) on value red (AMENDMENT items 2–5); and all 7 `maze-t-symmetric` rows (the six-seed condition table).
  - **Left deliberately:**
    - (1) **The w16 CMAZE-T per-arm generated states in `fig:cmazet-families`, `fig:cmazet-door-tint` and `fig:cmazet-deletion` (3 placeholders).** AMENDMENT-1 says "leave them" (figure work blocked elsewhere). ⚠ The amendment says **2**, but the draft has **3** of this kind; I left all three, since they share one cause.
    - (2) **NEW: the ladder's 32,768-row rival-set slot** (Ruling 3, "leave the column slot"). It is one row of `tab:ladder`.
- **TODO: 5 → 4.** Removed: "which difference between the two rival sets causes the gap" (the comparison it asked about is retired) and "no paired K\* ladder margin with CI" (now filled from w20: lowest lower bound +2.46). Added: **the abstract's CMAZE-T sentence as `\TODO` for the Head** (Ruling 12). The draft had no such TODO before; the asymmetric w16 sentence was still live, so I replaced it. The AI-use statement, the code archive and the pca_tensor byte-ratio TODOs are unchanged.

## Numbers verified before writing (a paraphrase is not a source)
- **Six-seed CMAZE-T mean = +0.0457 → +0.046.** This was recomputed from the six per-seed run JSONs (`verify/verify_maze.py`, `verify/verify_maze.out`). It confirms the Head's correction; +0.128 is wrong. Counts: wins on seeds 0, 2, 3; ties at the ceiling on seeds 1 and 4 (−0.0013, +0.0014; the CLU is within 0.0028 of the ceiling); loss on seed 5 (−0.1848).
- **Words-full TEST, against R\* (transformer 8k):** −0.0250 / −0.0013 / −0.0262 = Ruling 9. Against the per-seed best rival it is −0.026 / −0.001 / −0.026. Both are printed.
- **CLU-REL-A10:** seed 4 random_atoms = CLU = 0.9004; seed 5 transformer@80k = CLU = 0.9394. CLU − best +rel = −0.007 / +0.024 / +0.165 / +0.006 / 0 / 0. "97–100 %" holds on **seeds 3–4 only** (0.97 / 1.00). Seed 5 has no margin, and seed 2 is not closed. The text says exactly this.
- **Deletion (w20 JSON):**
  - (b) statistic deviation 0 on 9/9 legs.
  - ⚠ **AMENDMENT-1's rendered-deviation range "1.2e-09 … 7.8e-08" is wrong for the item scope.** The JSON gives 5.83e-10 / 9.74e-10 / 6.87e-10, so the true range is 5.8e-10 … 7.8e-08. Per-leg values are printed.
  - Book statistic 967,304,800 B. Book refit 2,367.3–5,912.9 s. Downdate − refit-without = 0.0 on 9/9 legs.
  - The conjunction refusal is printed as "undefined", never 0.
- ⚠ **A paraphrase error in the w20 advisor report.** It says "under red∧sphere the rivals lose < 1 novel/corner". The artefact shows the MLP at seed 2 going 43.174→40.220 and the continuous MLP at seed 2 going 26.844→17.275. The paper prints each rival before→after and says the store's loss is larger than every rival's change except the continuous MLP at seed 2.
- ⚠ **A result that was not in any report: the store's forgetting is not faster end to end.** The book refit takes 2,367–5,913 s, while a rival's retrain-without takes 547.7–2,064.5 s. The paper says so in §3.4 and in the limitations, and does not claim speed.

## What changed, ruling by ruling
1. **Headline ≈ +7.** This is in the abstract, contribution 1, §3.1 and Table 1: +7.03 (+7.49 / +8.60 / +5.02), every CI above +3.73, and +6.54 against the best fit of any length. The transformer margin is +10.468…+21.936. The rival search reaching the edge of the grid, and the MLP s0 still gaining +1.47 at 30×, are printed as a limitation.
2. **Retired from rendered text:**
   - T7's 16,384-row rows and margins, the "two rival sets" comparison, and fig_headline_steps (the step-matched diagnostic).
   - The three T7 rendered panels, whose arms are the 16k rivals.
   - The 16k retrain-without, NegGrad+ and fine-tune tables.
   - `learned_ref_ownpool` (2,560 steps) and the "+3.117" claim.

   Each keeps a LaTeX-comment artefact pointer.
3. **Shapes tables use the w20 set.** Table 1 = M-131k with DECLARED, 10× and 30× rows. The ladder has the data-matched set at 131,072 rows (MLP and continuous MLP also at 10×) plus a paired margin row, the earlier 32,256-row set, and the 32,768 slot.
4. **Deletion.**
   - `tab:del` is rebuilt with the four measurements on each scope×seed, (c) as a display, and a cost block.
   - A new **main-text paragraph**, "Deleting a conjunction changes what the book knows about the shared value", prints −6.13 / −6.43 / −6.30, the 20 corners at −32.1, and the random-1 % control at −0.02 / −0.11 / −0.01. The same limitation is also in the abstract, contribution 2 and limitations item 2.
   - The object-section definition now reads "exact on the store and its address book… not a claim that the answers on the surviving requests are unchanged".
5. **"Reads no training row"** appears only beside 0.97 GB / 967,304,800 B and 2,367–5,913 s, and is stated as "not a size or speed claim".
6. **`learned_1x` is removed.** This covers app:1x (the whole section), the K=5 1× opponents and bridge learner, and the 1× not-run bullets.
7. **K=5 appendix.** It is rebuilt from the w20 K=5 arm: rows 3–4 at 131,072 rows, and the ladder store rebuilt on 32,768 rows with all four controls at 0.000. It prints the one seed where that arm is behind (n=3, s2, −0.342). It also says why the old arm was replaced (random_atoms 1.24 / 2.03).
8. **All six CMAZE-T seeds are printed**, with a per-seed ceiling row: 3 wins, 2 ties at the ceiling, 1 loss, mean +0.046.
9. **Words-full TEST** is printed as behind by 0.0250 / 0.0013 / 0.0262.
10. **CLU-REL-A10** is printed with +rel beside every seed. The mechanism is named (relational request coding through the closed-form head). The seed-4 control tie and the seed-5 rival tie are stated. The text says "an input-representation result, not a store result" in §4.4, in the table, and in contribution 3.
11. **The six-seed condition table (`tab:cmazet-sym`)** replaces all 7 symmetric-input placeholders.
12. **The abstract's maze sentence is `\TODO`.**
- **New passages:**
  - (i) The measured per-seed ceiling, stated as a ceiling "for identity-coded readers… not a constant of the data", with a recommendation to report it per seed.
  - (ii) CMAZE-T as a minor contribution, explicitly "not as a benchmark".
- **Also:** C-512 is marked as inheriting the asymmetric scope and not promoted. The intro no longer claims measured approximate unlearning.

## Decisions taken (open choices; see LOG.md D1–D6)
- D1 treats all artefacts built on the 16k comparators as retired, including the approximate-unlearner table.
- D3 removes the "value still answered after deletion (32.2500)" row, which the Head's AMENDMENT-2 retired. It was in the abstract, contributions, limitations and ethics.
- D4 puts identical new prose into both drafts. Content identity is proven; the style draft's paraphrase of the replaced passages is lost in those regions.

## For the figures owner / next pass
- **`fig_ladder` panel (b) still draws the 2,560-step 32,768-row MLP.** Its caption prints the step count; replace it with the 32,768-row set when that set lands.
- **`fig_maze_conditions` (a)'s caption** still says A1 reads WIN on its 3-seed rule. That is accurate for panel (a), but the text now leads with six seeds.
- **The T7 per-arm panels** need a re-render against the data-matched rivals.

## What did NOT run
- No scored run, no compute beyond pdflatex/bibtex/gs, and no figure rebuild.
- The w16 per-arm CMAZE-T panels were not drawn (figure work).
- The 32,768-row ladder set is pending.
- There was no visual inspection of every page. Pages 1, 7, 10 and 40 were rendered and inspected; the deletion and condition tables were re-laid-out after that inspection.
- The main text is still ~18+ pp against ICLR's 9 pp (MF-11, the Head's), so the paper is still over length.
- No registry fold.
