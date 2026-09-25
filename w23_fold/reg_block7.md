
## ◈.21 WHAT CHANGED SINCE THE SIXTH PASS, AND WHAT THIS PASS READ

**Commissioned by** the w23 `fold` brief (`~/clu/tasks/w23/fold.md`, 2026-09-25; pre-authorised in full). **Base:** this file's own pass 6
(`N391`–`N397`, `◈.18`–`◈.20`; 8,415 lines, md5 `2df8597a…`). ⛔ Pass-1 to pass-6 entries are **not edited**. The only changes to them are **inserted**
`⟲ PASS 7` lines. Nothing was replaced, deleted or renumbered. This block adds **`N398`–`N405`**.

**Why a seventh pass. It is a timestamp gap, not a judgement call.** Pass 6 ended `2026-09-25T06:30:16Z`. `w22` seeds-and-tables pass 5 (branch
`w22/seeds5`, commit `3bb83da`) started `2026-09-25T06:17:43Z` and ended `2026-09-25T06:39:59Z`, after it. By pass 6's own rule (*"Anything that lands
after `2026-09-25T06:16:47Z` (this pass's START) is a new pass"*), its findings are a new pass. Pass 6's text mentions none of them.

⛔⛔ **Standing rule (Head, 2026-09-24): pre-registrations are banned.** Report everything, omit nothing. This pass writes no bar, label or prediction.
⭐ **A paraphrase is not a source** (`N351`, `N371`, `N381`, `N396`). Every number below is read from a JSON, a generated TABLES file, a draft line
(read only) or the curator's own arithmetic file (**CA7** = `~/clu/outputs/w23-fold/curator/curator_arithmetic_p7.json`). Where the w23 brief or a
spoke's prose disagrees with the artefact, **the artefact wins** and the disagreement is folded (`N404`).

**What this pass read** (unprefixed paths are relative to `~/clu/evidence/outputs/`):
- **`w22-seeds-and-tables5/`**: `FINDINGS.md`, `LOG.md`, `NEGATIVE-TESTS.txt`, `NEEDS-ARTEFACTS.json`, `SHA256-TABLES.txt`, START/END. The branch
  `w22/seeds5` @ `3bb83da`: `w22_seeds5/{negative_tests.sh, mutate.py, scan_draft.py}`. The register `~/clu/outputs/w18-seeds-and-tables/`
  (`TABLES.md`, `TABLES-w18.json`, `build_tables.py`, `cells_pass5.py`, `draft_pass3/pass2-manifest.json`, `draft_pass5/`).
- **Artefacts re-read for the numbers:** `w15-lace-poe-rows/GENTABLE-KSTAR.json` (`/PART_B_rows_3_4/arms/{PoE,LACE}/budget_per_seed`) and its
  `FINDINGS.md`; `w20-shapes-data-matched/SHAPES-MATCHED.json` (`/deletion_matched/per_scope/*/store/*/b/b2_book`); `w21-maze-t-seed4/MAZE-T-SEED4.json`.
- **The sweep of the w22 units no registry pass has named:** `~/clu/outputs/w22-draft-rulings/` (`FINDINGS.md`, `LOG.md`, `AMENDMENT-1.md`),
  `~/clu/outputs/w22-figures-redraw2/FINDINGS.md`, `~/clu/outputs/w22-paper-visuals/FINDINGS.md`; `~/clu/outputs/paper-visuals/cmazet_panels.py` (static
  read). Branches `w22/{draft-rulings, figures-redraw2, figures3, paper-visuals, registry6, seeds5}` listed with their tips.
- **The drafts, READ ONLY:** `~/clu/outputs/w17-paper/DRAFT-v2.tex` (md5 `6cecd36a…`) and `DRAFT-v2-style.tex` (md5 `72327303…`), byte-identical to
  pass 5's pinned copies. ⛔ **Nothing was written to `w17-paper/`, `figures/` or `paper-visuals/`. The `.tex` lock was not taken.**

**What this pass RAN (zero scored runs, zero fits):** the register build (twice, byte-identical), pass 5's negative-test suite, a stricter pass-7
suite, a reproduction of pass 5's harness defect, a differential re-test at the pass-3 and pass-4 register commits, a read-only coverage scan of the
drafts, and CA7. All in scratch copies, except the register install (`N402`).

## ◈.22 ⭐ THE ENTRIES — `N398`–`N405`

### N398 — ⭐⭐ **THE w15 FOLD IS CLOSED: 199/199 DRAFT NUMBERS UNDER `bar-d-score` / `coverage-control` / `lace-poe-rows` ARE REGISTER CELLS, CHECKED THREE WAYS (draft ↔ FINDINGS ↔ a named JSON leaf at the printed precision). Two earlier seeds-and-tables passes had declared it out of brief; pass 5 closed it. ⚠ SCOPE: this closes the THREE w15 SPOKES. It did not make every number the paper prints a register cell: before this pass, 212 of 2,293 draft tokens under a `% SRC` were.** · tier B (CURATOR-PROPOSED ◈) · claims hygiene / provenance (w22 seeds-and-tables pass 5, `TABLES.md` §13; ⭐ **the fold two passes had declared out of brief, now closed**)

**Source:** `w22-seeds-and-tables5/FINDINGS.md` §0; `~/clu/outputs/w18-seeds-and-tables/TABLES.md` §13; `w22_seeds5/scan_draft.py`; CA7 `register`;
`~/clu/outputs/w23-fold/curator/register_coverage_{before,after}.json`.

- **The check** (pass 5): each draft prints **199 number tokens** under a `% SRC` / `% SRCBLOCK` naming one of the three spokes. Each is three cells,
  re-checked on every build: (1) verbatim on its line of the pinned installed draft; (2) verbatim in the cited `FINDINGS.md` (**197/199**; the other 2
  are the corrected PoE value, `N399`); (3) a named JSON leaf at the printed precision (**199/199** after the correction). The 4 counts (`109/109`,
  `60 of 60`, `120 of 120`) are checked against every `corners_control_wins` leaf; 8 extra checks cover what the scanner cannot see. The `alone`
  probe margins are w8's, checked against the JSON w15's FINDINGS names. *"Nothing in the three spokes is left unfolded."*
- **Curator re-check (this pass):** the register built from pass 5's sources in scratch reproduces pass 5's `SHA256-TABLES.txt` exactly (`TABLES.md`
  `dc5dbde9…`, `TABLES-w18.json` `f51862c9…`); the live drafts are byte-identical to pass 5's pins.
- ⚠ **The scope, measured.** `w23_fold/register_coverage.py` (pass 5's token regex, every `% SRC` scope, not only w15) finds **2,293**
  (`DRAFT-v2.tex`) and **2,298** (`DRAFT-v2-style.tex`) number tokens under a SRC. Before this pass **212** of them in each draft were register draft cells
  (199 w15 + 13 w21). `w20-shapes-data-matched` alone carried 657 tokens and 0 cells; `w21-ladder-rivals-32k` 196 and 0.
  ⇒ *"The paper no longer prints values its register does not check"* (the w23 brief) holds **for the three w15 spokes only** (`N404` item 4). After
  this pass's fold of the W22-written numbers (`N402`): **1,308** of 2,293 and **1,309** of 2,298. **985** and **989** remain outside any draft cell,
  all on lines written before W22 (declared, `N405`).
- **Matrix row:** `CM-W22-2`.

### N399 — ⭐ **(provenance, the artefact wins) PoE's BUDGET WAS PRINTED 27.13× TWICE PER DRAFT; THE ARTEFACT SAYS 27.135. 14,226,660 / 524,288 = 27.13520050048828125, i.e. 27.14 at 2 dp: 27.13 is a TRUNCATION, copied from `w15-lace-poe-rows/FINDINGS.md`. Both drafts now print 27.135× (not 27.14, as the w23 brief says). ⚠ This registry carried the same 27.13× at `N360` and `CM-W15-3`; they are annotated ⟲, not edited.** · tier C (CURATOR-PROPOSED ◈) · provenance / numbers (w22 seeds-and-tables pass 5, edit E7; w15 `lace-poe-rows`)

**Source:** `w15-lace-poe-rows/GENTABLE-KSTAR.json` → `/PART_B_rows_3_4/arms/PoE/budget_per_seed/{0,1,2}/{x_store_reference, deployed_bytes, expert_params}`;
`w15-lace-poe-rows/FINDINGS.md` l.124 and l.158; `w22-seeds-and-tables5/FINDINGS.md` §0; CA7 `poe`, `lace`; drafts `DRAFT-v2.tex` l.1469 and l.1497 (read only).

| quantity | value | where |
|---|---|---|
| PoE `x_store_reference`, seeds 0 / 1 / 2 | 27.135 / 27.135 / 27.135 | JSON leaf |
| PoE deployed bytes (3,556,665 parameters × 4 B) | 14226660 | JSON leaf |
| 14,226,660 / 524,288, exact | **27.13520050048828125** | CA7 |
| to 2 dp (half-up; half-even gives the same) / truncated | **27.14** / 27.13 | CA7 |
| what w15's FINDINGS printed, and the drafts copied (twice each) | 27.13× | FINDINGS l.124, l.158 |
| what both drafts print now (edit E7; each line's SRC also names the JSON) | **27.135×** | `DRAFT-v2.tex` l.1469, l.1497; the style draft twice |

- ⚠ **Pass 5's own arithmetic line is slightly off:** *"14,226,660 / 524,288 = 27.1350"*. To 4 dp the quotient is **27.1352** (CA7). No printed number
  depends on it: 27.135 is the JSON leaf verbatim, and 27.14 is the 2-dp rounding either way.
- **LACE's multiples check out:** 31.08 (classifier only; exact quotient 31.07958221435546875) and 63.10 (with whitening; the JSON leaf is 63.095, exact
  63.09520721435546875). Both are correct roundings (CA7).
- A pass-5 negative test proves that reverting either draft cell to 27.13 fails the build (`w15_json_old_value`, re-run by the curator: caught).
- ⟲ at `N360` (this registry printed *"3,556,665 parameters = 27.13× the reference"*) and at `CM-W15-3` (*"27.13× the reference"*).

### N400 — ⚠⚠⚠ **(instrument integrity; a DEFECT CLASS) A NEGATIVE TEST THAT PASSES BY ACCIDENT VERIFIES NOTHING, AND IS WORSE THAN NO TEST. Pass 5's historical-tamper test "passed" on its first run by CRASHING on a missing output directory, and the cell it tampered was one the build never reads. It detected no tampering. The curator reproduced both halves: an UNMUTATED build into a missing directory exits 1 (a Traceback), and tampering an unread manifest cell builds rc=0, PASS. Fixed by pass 5; 12/12 negative tests now genuinely exit 1, each naming its target. ⭐ This programme relies on such tests: every "negative test exits 1" in its record is only as good as the harness that counted it.** · tier A (CURATOR-PROPOSED ◈) · process / instrument integrity / claims hygiene (w22 seeds-and-tables pass 5 `LOG.md` and `FINDINGS.md` §3; curator reproduction; ⭐ **folded as its own entry, by the brief**)

**Source:** `w22-seeds-and-tables5/LOG.md` (the register line: *"historical-tamper test fixed: it had "passed" by a crash on a missing out dir and
tampered a non-historical cell; Traceback no longer counts as caught"*); `FINDINGS.md` §3; `NEGATIVE-TESTS.txt`; `w22/seeds5:w22_seeds5/{negative_tests.sh,
mutate.py}`; curator: `~/clu/outputs/w23-fold/curator/{NEGATIVE-TESTS-pass5-rerun-on-pass5-register.txt, NEGATIVE-TESTS-pass7.txt}`, `LOG.md`; CA7 `pass2_manifest`.

**The defect, as pass 5 recorded it.** On the first run, (a) the build's output directory did not exist, so `build_tables.py` crashed while writing
`TABLES.md`, exiting non-zero; the harness counted any exit 1 as a catch. (b) The tampered cell was not one the build reads from the pass-2 manifest.
Either alone makes the test vacuous. ⚠ The first-run harness is not committed; only the fixed one is. The defect is known from the LOG and FINDINGS.

**Reproduced by the curator (scratch copies of the pass-5 register; nothing installed):**

| probe | what it is | result |
|---|---|---|
| (a) | the UNMUTATED register, built into a directory that does not exist | **rc=1**, `FileNotFoundError` Traceback. ⇒ a harness that counts rc≠0 as "caught" passes every test with no mutation at all |
| (b) | tamper `a1_audit`, a pass-2 manifest cell the build does not read | **rc=0, PASS.** The build reads the manifest for its **29** historical cells only; **478 of 507** manifest cells are never read (CA7). Tampering any of them is undetectable by design |
| fixed harness | `mkdir -p "$d/out"`; a Traceback is never a catch; the tamper targets `sorted(cells_historical)[0]` = `b_1012` | re-run by the curator: **12/12 caught**, and each failure message names the mutated target (`p4_c0_b`, `d4_index`, `p5_w15d_1`, `p5_w15f_1`, `p5_w15j_116`, `p5_w20h_1`, `p5_tie4_15`, `p5_tie5_1`, the two pointer checks, the bare number, `b_1012`) |

**Retro-check of the earlier passes' negative tests.** Passes 2, 3 and 4 of the register report negative tests "all exit 1" (`w18-seeds-and-tables/LOG.md`),
pass 3 including a tampered historical cell. **None of those harnesses was committed as a script**, so what they exercised cannot be recovered.
- ⚠ **At both older states a plain exit-code test is now meaningless:** the unmutated pass-3 (`f4f7c3d`) and pass-4 (`f52e794`) registers exit 1 today
  (12 and 15 `CELL FAIL`s: their live-draft cells point at text the drafts no longer hold). Any mutation "passes".
- ⭐ **A differential test still works:** tampering `b_1012` at each commit adds **exactly one** failure, `CELL FAIL b_1012`, to the untampered failure
  set. ⇒ The historical guard itself did detect tampering at passes 3 and 4. Whether their recorded tests exercised it cannot be shown.

**Residual weakness in the FIXED pass-5 harness** (not a false result; all 12 messages were checked by hand): it does not require (0) that the unmutated
copy builds rc=0 in the same run, nor (3) that the failure names the mutated target; a rc=1 without a Traceback, for any reason, counts.
⭐ **The pass-7 suite (`w23_fold/negative_tests7.sh`) adds both, plus a self-test:** baseline rc=0 first (else every test is VOID, exit 99); caught =
rc 1 AND no Traceback AND the output names the target; and a crash on a missing directory must be reported NOT CAUGHT. **7/7 caught; self-test OK.**

**The defect class, stated for reuse** (curator's reading; the Hub decides whether it becomes a rule, `§I8.23` flag 2): a negative test is evidence
only if (i) the unmutated baseline passes in the same harness, (ii) the mutation is one the checker actually reads, (iii) the exit is the checker's
own failure, not a crash, and (iv) the failure names the thing mutated.
- ⭐ **Two w22 siblings of the same class, found in the sweep (`N403`):** `figures-redraw2`'s per-tuple reader assumed `[correct, n]` where the replay
  stores `[n, correct]`, and **its own gate caught it** (a check that worked). `paper-visuals/build.sh` "rebuilds" only when run from a different
  directory: in place it exits 1 with 0 figures (a check whose pass depends on where it runs).
- **Matrix row:** `CM-W22-3`.

### N401 — ⭐ **THE LADDER'S 32,768-ROW TABLE PLACEHOLDER IS FILLED; THE w16 CMAZE-T PER-ARM PLACEHOLDERS ARE PERMANENTLY BLOCKED, AND THERE ARE THREE, NOT TWO. w16 persisted no arm object and no per-transition prediction, so no artefact anywhere can fill them; only a re-run of w16 could.** · tier C (CURATOR-PROPOSED ◈) · figures / placeholders (w22 seeds-and-tables pass 5, edits E3 and E5; `NEEDS-ARTEFACTS.json`)

**Source:** `w22-seeds-and-tables5/FINDINGS.md` §2, §4; `NEEDS-ARTEFACTS.json`; `~/clu/outputs/w19-success-failure-panels/LOG.md:2`; CA7 `DRAFT-v2.tex`,
`DRAFT-v2-style.tex` (read only).

- **Filled (edit E3):** `tab:ladder`'s store-content block (32,768 rows; 5,120 and 51,200 steps; eight family rows; store − strongest, paired and
  step-matched, with means), copied by script from `w21-ladder-rivals-32k/TABLES.md` (P1, L3). Its numbers are now register cells (`N402`). Folded as
  results at `N393`.
- **Blocked:** `fig:cmazet-families`, `fig:cmazet-door-tint`, `fig:cmazet-deletion`. `\PLACEHOLDER{` uses in the installed drafts: **3 each**
  (`DRAFT-v2.tex` l.2621, l.2646, l.2658; style l.2612, l.2637, l.2649; CA7). `\TODO{` uses: **4 each** (the Head's abstract CMAZE-T sentence; the
  `pca_tensor` byte ratio; the AI-use statement; the code archive). None is a table cell.
- ⚠ **"Two" is a paraphrase that has now appeared three times:** Head AMENDMENT-1 (*"leave them"*, counted as two; flagged by `w22-draft-rulings`),
  pass 5's own brief, and the w23 brief. The drafts carry **three** of this kind, with one cause (`N404` item 2).
- ⚠ **Not the same slot:** the **figure** slot for the 32,768-row ladder set is still open. `fig_ladder` panel (b) still draws the 2,560-step
  32,768-row MLP (flagged by draft-rulings and pass 5; `§I8.21` flag 4 stands). ⭐ Verified read-only: the drafts place **14** `\figslot`s each, among
  them `fig_cmazet_rel_per_arm` (figures3's REL panels). `fig_headline_steps` and the `rc_w19_t7_*` panels are not placed (`§I8.21` flag 6, verified).
- **Matrix row:** `CM-W22-4`.

### N402 — ⭐⭐ **(register, TABLES.md §17) THE NUMBERS W22 WROTE INTO THE DRAFTS ARE NOW REGISTER CELLS: 1,094 (`DRAFT-v2.tex`) and 1,095 (style) tokens, each a DRAFT cell on the pinned installed draft and a SOURCE cell on the first file its `% SRC` names. 2 tokens per draft have no verbatim line in the cited file, and are printed as findings. ⚠ A source cell proves PRESENCE in the cited file, not meaning; pass 5's JSON-leaf check is not claimed.** · tier B (CURATOR-PROPOSED ◈) · claims hygiene / provenance (w23 fold; the w22 draft-rulings, figures-redraw2, figures3 and seeds5 draft edits)

**Source:** `~/clu/outputs/w18-seeds-and-tables/{TABLES.md §17, cells_pass7.py, build_tables.py, TABLES.pass7.frag.md}`; `w23_fold/{gen_cells_pass7.py,
make_template7.py, stage_register7.py, negative_tests7.sh, verify_insert_only.py}`; CA7 `register`; `~/clu/outputs/w23-fold/curator/{gen_stats.json,
unsourced.json, verify_register7.json, install-build.txt}`.

- **Scope:** every number token on a line of the pinned pass-5 draft that is not in the pinned pass-4 text (difflib hunks: **674** lines in
  `DRAFT-v2.tex`, **714** in the style draft), under a SRC, not already a pass-5 draft cell. These are the lines W22 wrote.
- **Result:** **3,024** new cells; the register now verifies **5,306** cells verbatim (was 2,282), 299 JSON cells, 3 JSON equalities, 0 unsourced
  template numbers. **827** distinct source cells in **30** source files. Of the 1,094 `DRAFT-v2.tex` tokens, **992** carry 4+ digits; **102** are
  **short** (fewer than four digits), where presence in the file is weak evidence. Each row says which.
- **The 2 tokens per draft with no verbatim line in the cited file** (`TABLES.md` §17.3, as cells):
  1. **`0.6407`** (the easy/hard cut). The line cites `MAZE-T-SEED4.json`, whose leaf is `0.6407297594090047`; the rounded value is verbatim only in an
     **uncited** file, `w21-maze-t-seed4/TABLES.md` l.11. A correct rounding; the SRC should also name that file.
  2. **`16 of 16`** (*"the random-atoms control and the best relational rival equal the store on 16 of 16 tuples"*). The cited
     `w21-rel-newseeds/FINDINGS.md` l.62 carries the control half (*"random_atoms matches the CLU exactly on all 16 held-out tuples of seed 4"*). The
     **relational-rival half is not in it**. It rests only on `w22-figures-redraw2` (*"The best +rel rival (transformer, 8k and 80k) also equals it on
     16/16"*), not cited on that line. Pass 6's re-count (`N392`) covers the control half only.
  ⇒ Both are draft-owner fixes to a `% SRC` comment. ⛔ Not edited here.
- **Acceptance:** staged in scratch, two builds byte-identical, then installed; the installed build is byte-identical to both (`TABLES.md` sha256
  `b53d55c574a18ed5…`). difflib against pass 5: `TABLES.md` **2,228** lines inserted, `TABLES.template.md` **2,228**, `build_tables.py` **10**;
  **0 replaced, 0 deleted** in each. Pass 5's 12 negative tests on the new register: 12/12 caught. The pass-7 suite: 7/7 caught, self-test OK (`N400`).
- ⚠ **One design choice, recorded:** the pass-7 cells are plain verbatim cells in a new `cells_pass7.py`; `build_tables.py` gains two inserted blocks
  (import, merge) and no changed line.
- **Matrix row:** `CM-W22-2`.

### N403 — ⚠ **(the sweep) THREE w22 UNITS NO REGISTRY PASS HAD NAMED: `draft-rulings`, `figures-redraw2`, `paper-visuals`. Most of their numbers were already folded from the underlying w20/w21 artefacts. What was NOT folded: AMENDMENT-1's rendered-deviation range is wrong for the item scope; the Head's AMENDMENT-2 retirement of the "value still answered after deletion (32.2500)" row; the 1,280-step comparator count is DERIVED, stored in no artefact field; and `paper-visuals/build.sh` does not rebuild in place. ⭐ Nothing else in `w22-*` is outstanding.** · tier C (CURATOR-PROPOSED ◈) · provenance / figures / process (w22 `draft-rulings`, `figures-redraw` pass 2, `paper-visuals`)

**Source:** `~/clu/outputs/w22-draft-rulings/{FINDINGS.md, LOG.md, AMENDMENT-1.md}`; `~/clu/outputs/w22-figures-redraw2/FINDINGS.md`;
`~/clu/outputs/w22-paper-visuals/FINDINGS.md`; `w20-shapes-data-matched/SHAPES-MATCHED.json`; `~/clu/outputs/paper-visuals/cmazet_panels.py` l.19,
`cmazet_extract_rel.py` l.28 (static read); CA7 `rendered_dev`.

**NEW here (verified against the artefact, or against the registry's own curator arithmetic):**
1. ⚠ **AMENDMENT-1's (b) range is wrong for the item scope.** The amendment says the rendered-feature relative deviation is *"1.2e-09 … 7.8e-08"* on all
   three scopes. `SHAPES-MATCHED.json` gives, over the 9 legs, min **5.829167148974079e-10** (item, seed 0) and max **7.812795388417726e-08**
   (conjunction, seed 1). All three item legs are below 1.2e-09: 5.829167148974079e-10 / 9.74098579575866e-10 / 6.873774882620864e-10. draft-rulings found
   this and printed per-leg values; no registry pass had it. (b) remains exact within its declared bar (`N381` item 1).
2. **Retired from rendered text by the Head's AMENDMENT-2:** *"the value still answered after deletion (32.2500)"* (draft-rulings D3; the abstract,
   contributions, limitations and ethics). Also retired (D1): every artefact built on the 16,384-row comparators, including the approximate-unlearner
   table. The retired-number grep of the rendered PDFs finds 0 hits, with a positive control on the pre-edit PDF.
3. **The draft's 16,384-row comparators' 1,280 steps are derived** (20 epochs × ⌈16,384/256⌉ batches; `figures-redraw2` discrepancy 2). No artefact
   stores it as a field. `N373` quotes 1,280 without saying so (⟲ added).
4. ⚠ **`paper-visuals/build.sh` does not run in place** (`paper-visuals` §3): `REPO = dirname(HERE)` resolves to `~/clu/outputs`, so the in-place rebuild
   exits 1 with 0 figures; the committed copy in a worktree rebuilds 27/27 byte-identical. Still present on 2026-09-25 (static read). A figures-owner fix.
5. ⚠ **draft-rulings' "+0.128 is wrong"** (FINDINGS; LOG: *"Head's correction CONFIRMED; +0.128 is wrong"*). It is wrong only as a value of the
   per-seed-best estimand, whose six-seed mean is +0.0457. As the R\*-by-mean margin it is right: 0.12802168076341014 (`N385`, CA5). The drafts print
   +0.046 by the Head's correction; that is not questioned here (`N404` item 6).
6. **Editorial, for the paper Hub** (`paper-visuals` §5): figures 6, 8 and 9 (ladder, K-curve, deletion) were `\ref`'d only from the placeholder index.
   Not re-checked against the current drafts at this pass.

**ALREADY FOLDED (pointer only):** the store's forgetting is not faster end to end (book refit 2,367–5,913 s against a rival's 547.7–2,064.5 s:
`N375`, `N376`); the six-seed CMAZE-T mean +0.046 (`N385`); words-full against R\* (`N382`); the seed-4 and seed-5 ties (`N392`); "97–100 %" holds on
seeds 3–4 only and seed 5 is 0/0 (`N392`, share n/a); the w20 advisor report's "rivals lose < 1" (`N381` item 2); figures-redraw2's gate 2, not run
there for want of `w21_clu`, later passed by figures3 (`N395`); the A1 seed-1 and seed-5 rival-above-CLU marks (`N384`).
- **Matrix row:** `CM-W22-5`.

### N404 — ⚠ **(provenance) SIX PLACES WHERE THE w23 BRIEF, PASS 5's PROSE OR A w22 SPOKE'S PROSE DISAGREES WITH THE ARTEFACT (one of them, item 5, holds and is only qualified). The artefact wins each one. `N351` / `N371` / `N381` / `N396`'s rule applied a fifth time, three times to this pass's own brief.** · tier C (CURATOR-PROPOSED ◈) · process / provenance (the w23 brief; `w22-seeds-and-tables5` FINDINGS and LOG; `w22-draft-rulings` FINDINGS and LOG)

| # | the paraphrase | the artefact | where |
|---|---|---|---|
| 1 | Brief: PoE *"⇒ 27.14"* (read as what the drafts now print) | The drafts print **27.135×** (the JSON leaf verbatim). 27.14 is the correct 2-dp rounding, printed nowhere | `N399`; drafts l.1469, l.1497 |
| 2 | Brief: *"the two w16 CMAZE-T per-arm placeholders"* | **Three** `\PLACEHOLDER{` uses per draft, one cause (also AMENDMENT-1's miscount) | `N401`; CA7 |
| 3 | Pass 5: *"14,226,660 / 524,288 = 27.1350"* | **27.1352** to 4 dp (27.13520050048828125) | `N399`; CA7 |
| 4 | Brief: *"the paper no longer prints values its register does not check"* | True for the three w15 spokes (199/199). Whole drafts: **212 of 2,293** tokens were draft cells; after this pass 1,308; 985 pre-W22 tokens remain unchecked by a live draft cell | `N398`, `N402` |
| 5 | Brief: *"12/12 negative tests now genuinely exit 1"* | ⭐ **Holds**, re-run by the curator, and each message names its target. ⚠ The fixed harness does not itself check the baseline or the target (`N400`) | `N400` |
| 6 | draft-rulings: *"+0.128 is wrong"* | Wrong only for the per-seed-best estimand (+0.0457). **0.12802168076341014 is the R\*-by-mean margin** (CA5), a different estimand, not an error. The Head chose which one the paper prints | `N403` item 5; `N385` |

- **Checked and consistent (not disagreements):** pass 5's LOG says *"197/199"* for the JSON check and FINDINGS says *"199/199"*: the LOG line precedes
  the PoE correction, FINDINGS follows it. The install check in the LOG (*"w15 check 199/199 (2 FINDINGS-miss = the corrected 27.135 by design)"*)
  agrees with FINDINGS.

### N405 — ⚠ **(process, omit nothing) THE PASS-7 SWEEP AND ITS DECLARED NOT-RUNS. ⛔ The PRNG-replicate rival fits (pass 6's ask, `N394`) are DECLARED NOT RUN by the Head's ruling: "not cheap enough". Recorded here so the request is not lost. ⭐ Nothing else from w22 is outstanding.** · tier C (CURATOR-PROPOSED ◈) · process / claims hygiene (the w23 brief; the Head's ban)

| item | status |
|---|---|
| ⛔ **PRNG-replicate rival fits** at the carried budgets (k replicates of the carrying rival per seed; Shapes rows 3–4 and CMAZE-T), the measurement `N394` says would put a refit interval beside every per-corner CI | ⛔ **DECLARED NOT RUN, the Head's ruling (w23 brief): "not cheap enough".** The `N394` caveat stands as printed; `§I8.21` flag 1 is answered by this ruling (`§I8.23` flag 1) |
| ⛔ the three w16 CMAZE-T per-arm panels | **PERMANENTLY BLOCKED** (`N401`): no artefact exists; only a re-run of w16 |
| ⛔ the numbers the drafts print on lines written before W22 (**985** in `DRAFT-v2.tex`, **989** in the style draft) | **NOT FOLDED, declared.** Outside this brief (fold what W22 left). They remain checked by their writers' provenance checks, and pass-2 to pass-4 draft cells check the pinned historical text, not the live drafts. `§I8.23` flag 3 |
| ⚠ a JSON-leaf (third) check for the pass-7 cells | **not done**: source cells prove presence only (`N402`) |
| ⚠ the `% SRC` fixes for `0.6407` and `16 of 16`; `paper-visuals/build.sh`; `fig_ladder` panel (b) | owed by the draft and figures owners; ⛔ not edited here |
| ⛔ any scored run, fit, GPU, ledger write, PREREG, bar or label | **ZERO**. CA7 reads JSONs and divides; the register build and negative tests check text |
| ⛔ any write to `w17-paper/`, `figures/`, `paper-visuals/`; the `.tex` lock | **NOTHING WRITTEN; lock not taken.** Drafts read to check numbers only |

⭐ **The sweep, stated plainly:** of the w22 evidence and branches (`w22-seeds-and-tables5`, `w22-registry-fold6`, `w22-figures3-FINDINGS.md`; outputs
`w22-{draft-rulings, figures-redraw2, figures3, paper-visuals, registry-fold6, seeds-and-tables5}`; branches `w22/{draft-rulings, figures-redraw2,
figures3, paper-visuals, registry6, seeds5}`), every finding is now folded (`N391`–`N405`) or declared above. **Nothing else is outstanding.**

## ◈.23 ⭐ THE FOLD-IN LEDGER, PASS 7 — every brief item, and where it now lives

| brief item | status | where |
|---|---|---|
| ⭐⭐ the w15 fold CLOSED, 199/199 at printed precision; earlier passes had declared it out of brief | ⭐ **FOLDED**, ⚠ with its scope measured (the three w15 spokes; 212 of 2,293 draft tokens were cells) | `N398`; `CM-W22-2` |
| ⭐ PoE 27.13× → 27.14 (the artefact wins) | ⭐ **FOLDED**, ⚠ the drafts print **27.135×** (`N404` item 1); pass 5's "27.1350" is 27.1352 | `N399`; ⟲ at `N360`, `CM-W15-3` |
| ⭐⭐ the historical-tamper test passed by crashing; fixed; 12/12 | ⭐ **FOLDED AS A DEFECT CLASS**, reproduced by the curator, retro-checked at passes 3–4, and a stricter suite added | `N400`; `CM-W22-3` |
| ladder 32,768-row placeholder filled; w16 per-arm placeholders PERMANENTLY BLOCKED | ⭐ **FOLDED**, ⚠ **three** placeholders, not two | `N401`; `CM-W22-4` |
| ⭐ also check `w22-*` and the w22 branches | ⭐ **DONE**: three unnamed units folded; the W22-written draft numbers made register cells; **nothing else outstanding** | `N402`, `N403`, `N405` |
| ⛔ PRNG-replicate rival fits | ⛔ **DECLARED NOT RUN**, the Head's reason: "not cheap enough" | `N405`; ⟲ at `N394`, `CM-W21-6` |
| `TABLES.md` only where a cell is genuinely missing | ⭐ §17 only (the W22-written numbers); pass 5's cells not restated | `N402` |

⭐ **No eighth pass is required for these items.** Anything that lands after `2026-09-25T09:17:46Z` (this pass's START) is a new pass.
