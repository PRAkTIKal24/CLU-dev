# w23 spoke/fold — FINDINGS (batmac, doc-curator; registry pass 7, register pass 6)

**Scored runs: 0. Compute: 0.** Nothing was fitted or scored, and nothing was written to a ledger. No PREREG, bar or label.
**⛔ Nothing was written to `w17-paper/`, `figures/` or `paper-visuals/`. The `.tex` lock was not taken.** The drafts were read only, to check numbers; they are byte-identical to pass 5's pins (md5 `6cecd36a…` / `72327303…`).
**Background jobs: none. Sub-agents: none.** Nothing was killed; tmux was not touched.

## 0. What shipped
| file | before → after | what this pass adds |
|---|---|---|
| `~/clu/outputs/w21-registry-fold/negative_results.md` | 8,415 → **8,672** lines (8,416 → 8,673 counting the final newline) | **`N398`–`N405`**; `◈.21`–`◈.23`; a header note; `⟲ PASS 7` lines at `N360`, `N373`, `N394`, `N397`, and after `◈.20` |
| `~/clu/outputs/w21-registry-fold/claims_matrix.md` | 2,439 → **2,521** lines, **v2.28 (PROPOSED)** | **`§I8.22`**: `CM-W22-2`…`CM-W22-5`, `CM-W17-10`; **`§I8.23`**: 5 open flags; `⟲ PASS 7` lines at `CM-W15-3`, `CM-W21-6`, `§I8.21` flags 1 and 6. The v2.28 title is inserted above the v2.27 title, which is kept verbatim |
| `~/clu/outputs/w18-seeds-and-tables/` (the register) | 2,282 → **5,306** cells | **§17**: the numbers W22 wrote into the drafts, as cells; `cells_pass7.py`; `build_tables.py` +10 inserted lines; FINDINGS / LOG notes (insert-only) |
| `~/clu/outputs/w23-fold/` | new | this file, `LOG.md`, copies of the registry and matrix, `curator/` (CA7, coverage, verify, audit, negative tests, bases) |

**Acceptance, proved by script** (`curator/verify_pass7{,_installed}.json`, `audit_pass7{,_installed}.json`, `rounding_check_p7.json`, `verify_register7.json`, `verify_register_docs.json`):
- **difflib against pass 6: registry 257 lines inserted, matrix 82; 0 replaced, 0 deleted.** The base files are an exact subsequence of the new ones. Re-verified on the installed copies.
- **Headings:** 397 `### N` headings, 0 duplicates; `N398`–`N405` each appear once. 72 `CM-` rows, 0 duplicates. No duplicate `◈.` / `§I8.` section.
- **Number audit:** every decimal and every 4+-digit token in the inserted lines appears verbatim in a source file, not glued to a preceding digit. A strict pass checks each entry and hunk against **only the files it names** (plus CA7 and this pass's own record). **0 failures.** 0 tokens occur only as the prefix of a longer number.
- **Register:** `TABLES.md` and `TABLES.template.md` 2,228 lines inserted each, `build_tables.py` 10, register FINDINGS 10, register LOG 1; **0 replaced, 0 deleted**. Two scratch builds and the in-place build are byte-identical (`SHA256-TABLES-pass7.txt`).
- Base md5 values were re-checked immediately before each install.

## 1. The known gap (seeds-and-tables pass 5, END 06:39:59Z, after registry pass 6's END 06:30:16Z), folded
- ⭐⭐ **The w15 fold is CLOSED** (`N398`, `CM-W22-2`). 199/199 draft numbers under `bar-d-score` / `coverage-control` / `lace-poe-rows` are three-way register cells. Two earlier passes had declared this out of brief; it is now recorded as closed.
  - The curator rebuilt pass 5's register in scratch; both sha256 values equal pass 5's record.
  - ⚠ **Scope, measured:** that closes the three w15 spokes only. Before this pass, **212 of 2,293** number tokens under a `% SRC` in `DRAFT-v2.tex` were register draft cells. *"The paper no longer prints values its register does not check"* (brief) is true only for those three spokes.
- ⭐ **PoE 27.13× was wrong** (`N399`). The JSON leaf is 27.135; 14,226,660 / 524,288 = 27.13520050048828125, which is 27.14 at 2 dp, so 27.13 was a truncation.
  - ⚠ **The drafts print 27.135×, not 27.14** as the brief says.
  - ⚠ Pass 5's "= 27.1350" is 27.1352 to 4 dp.
  - This registry printed 27.13× at `N360` and `CM-W15-3`. Both are annotated ⟲; neither is edited.
- ⚠⚠⚠ **The defect class: a negative test that passes by accident** (`N400`, `CM-W22-3`). **Reproduced by the curator**, in scratch:
  - An **unmutated** build into a missing directory exits 1 with a Traceback. A harness that counts rc≠0 as "caught" therefore passes with no mutation at all.
  - Tampering a manifest cell the build never reads gives **rc=0, PASS**. 478 of the 507 manifest cells are never read.
  - Pass 5's fixed suite: **12/12 caught**, and the curator checked that each message names its target.
  - **Retro-check:** the pass-2 to pass-4 negative tests were never scripted, so they cannot be re-verified. At the pass-3 and pass-4 commits, the unmutated registers now exit 1, so any plain rc test "passes" there. A differential test shows the historical guard itself worked (it adds exactly `CELL FAIL b_1012`).
  - ⭐ **New stricter suite** (`w23_fold/negative_tests7.sh`): the baseline must pass first; a catch needs rc 1, no Traceback, AND the target's name in the output; a self-test checks that a crash is NOT counted as a catch. **7/7, self-test OK.**
  - A proposed four-criterion rule is left for the Head (`§I8.23` flag 2).
- ⭐ **Placeholders** (`N401`, `CM-W22-4`):
  - The ladder's 32,768-row **table** slot is filled.
  - The w16 CMAZE-T per-arm panels are **permanently blocked**. ⚠ There are **three** (`\PLACEHOLDER{` ×3 per draft), not two as the brief and AMENDMENT-1 say.
  - The ladder **figure** panel (b) is still open (figures owner).

## 2. The sweep ("also check"): what else was unfolded, and where it now is
- ⭐⭐ **The numbers W22 wrote into the drafts had no register cell. They are now in TABLES.md §17** (`N402`). This was the one genuinely missing TABLES content.
  - **Scope:** every token on the lines W22 wrote (674 lines in `DRAFT-v2.tex`, 714 in the style draft).
  - **Cells:** 1,094 / 1,095 tokens. Each becomes a DRAFT cell (pinned installed draft) and a SOURCE cell (the first file its `% SRC` names).
  - **Result:** 3,024 new cells. Live draft-cell coverage rose from **212 to 1,308 of 2,293** (`DRAFT-v2.tex`) and from 212 to 1,309 of 2,298 (style).
  - ⚠ A source cell proves **presence** in the cited file, not meaning. 102 of the 1,094 `DRAFT-v2.tex` rows are short tokens (fewer than 4 digits), and each row is marked.
  - ⚠ **2 tokens per draft have no verbatim line in the cited file.** Both are printed as cells in §17.3:
    - `0.6407` is a correct rounding of the cited JSON leaf. The rounded value appears verbatim only in an uncited file.
    - For `16 of 16`, the relational-rival half rests only on `figures-redraw2`, which the line does not cite.
    - Both are `% SRC` fixes for the draft owner. Not edited here.
- ⚠ **Three w22 units no registry pass had named** (`N403`, `CM-W22-5`): `draft-rulings`, `figures-redraw2`, `paper-visuals`. New items, each verified:
  1. **AMENDMENT-1's "1.2e-09 … 7.8e-08" is wrong for the item scope.** `SHAPES-MATCHED.json` gives a 9-leg range of 5.829167148974079e-10 … 7.812795388417726e-08.
  2. **The 32.2500 row was retired** by the Head's AMENDMENT-2.
  3. **The 1,280 comparator steps are derived.** No artefact field stores them (⟲ at `N373`).
  4. **`paper-visuals/build.sh` does not rebuild in place.** The defect is still present (static read).
  5. ⚠ **draft-rulings' "+0.128 is wrong"** is wrong only for the per-seed-best estimand. **0.12802168076341014 is the R\*-by-mean margin** (CA5, `N385`), which is a different estimand, not an error.
  - Everything else in those units was already folded. `N403` lists each item with its pointer.
- **Provenance** (`N404`): six places where the brief or a spoke's prose disagrees with the artefact. Three of them are in this pass's own brief: 27.14, "two" placeholders, and "the paper no longer prints…". The artefact wins each one.
- ⭐ **STATED PLAINLY: after this pass, NOTHING ELSE from w22 is outstanding.** The sweep covered:
  - evidence: `w22-seeds-and-tables5`, `w22-registry-fold6`, `w22-figures3-FINDINGS.md`;
  - outputs: `w22-{draft-rulings, figures-redraw2, figures3, paper-visuals, registry-fold6, seeds-and-tables5}`;
  - branches: `w22/{draft-rulings, figures-redraw2, figures3, paper-visuals, registry6, seeds5}`.

  Every finding in them is now in `N391`–`N405`, or is declared in `N405`.

## 3. Declared not run / not done (omit nothing)
- ⛔ **The PRNG-replicate rival fits** (registry pass 6's ask, `N394`) were **NOT RUN**. The Head ruled them *"not cheap enough"*. This is recorded at `N405`, `CM-W17-10` and `§I8.23` flag 1, with ⟲ at `N394` and `CM-W21-6`, so the request is not lost.
- ⛔ **The draft numbers on lines written before W22 are not folded into the register:** 985 in `DRAFT-v2.tex` and 989 in the style draft. They are outside this brief and are declared at `N405` and `§I8.23` flag 3.
- ⛔ **No JSON-leaf (third) check for the pass-7 cells.** They check presence only.
- ⛔ **Not edited:** any draft, figure, `paper-visuals` file or evidence file. That includes the `% SRC` fixes and `build.sh`; they are owed by their owners.
- ⛔ **No other checker's negative tests were audited** (provenance, style-diff, figure gates). This is `§I8.23` flag 2.
- **No `NEEDS-ARTEFACTS.json` was written.** Nothing this brief needed is missing. The only unrecoverable object is pass 5's first-run (broken) harness, which was never committed; the defect is known from pass 5's LOG and FINDINGS, and the curator reproduced it.
- ⚠ The tiers of `N398`–`N405` are CURATOR-PROPOSED, and v2.28 is PROPOSED.
- ⭐ **No eighth pass is required for these items.**
