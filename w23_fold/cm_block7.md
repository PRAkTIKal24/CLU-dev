
## §I8.22 ◈◈◈◈◈◈◈ THE PASS-7 ROWS: the w15 fold closed (and its scope), the W22 draft numbers made register cells, PoE 27.135×, the negative-test defect class, and the w22 sweep

⛔ This section is **inserted**. No earlier row is edited. The `⟲ PASS 7` lines above are inserted as well. Registry entries: `N398`–`N405`. All tiers
are CURATOR-PROPOSED. ⛔⛔ No PREREG, no bar, no label. Nothing was written to `w17-paper/`, `figures/` or `paper-visuals/`; the drafts were read only.

### **CM-W22-2** ⭐⭐ THE REGISTER AND THE PAPER: **the w15 fold is CLOSED (199/199, three ways); the W22-written numbers are now cells (1,094 / 1,095); 985 / 989 pre-W22 draft tokens are still outside any live draft cell**
- **Claim, as the artefacts support it:** every number the drafts print under the three w15 spokes (`bar-d-score`, `coverage-control`, `lace-poe-rows`)
  is a register cell checked draft ↔ FINDINGS ↔ JSON leaf at the printed precision (`N398`). Every number W22 wrote into the drafts is a register cell
  checked draft ↔ cited source file, by presence (`N402`). Of 2,293 (`DRAFT-v2.tex`) number tokens under a `% SRC`, 1,308 are now live draft cells
  (212 before this pass).
- ⛔ **FORBIDDEN:** *"every number in the paper is checked by the register"* (985 / 989 pre-W22 tokens are not, `N405`); *"the paper no longer prints
  values its register does not check"* without the scope "for the three w15 spokes" (`N404` item 4); a pass-7 source cell quoted as a semantic check
  (it proves presence; 102 of the 1,094 `DRAFT-v2.tex` rows are short tokens).
- **Owed to the draft owner (not edited here):** the `% SRC` of the `0.6407` line should also name `w21-maze-t-seed4/TABLES.md`; the `16 of 16` line
  should cite `w22-figures-redraw2` for the relational-rival half (`N402`).
- **Registry:** `N398`, `N402`, `N404`, `N405`.

### **CM-W22-3** ⚠⚠⚠ THE NEGATIVE-TEST DEFECT CLASS: **a test that passes by accident verifies nothing, and is worse than no test**
- **Claim, as the artefacts support it:** pass 5's historical-tamper test first "passed" by crashing on a missing output directory, on a cell the build
  never reads. The curator reproduced both halves (unmutated build → rc=1 with a Traceback; unread-cell tamper → rc=0 PASS; 478 of 507 manifest
  cells are unread). The fixed suite catches 12/12, each naming its target (`N400`).
- ⚠ **Reach:** the pass-2, pass-3 and pass-4 register negative tests were never scripted; their "exit 1" records cannot be re-verified. A
  differential re-test shows the historical guard itself worked at passes 3 and 4.
- ⭐ **Proposed rule (the Hub's call, `§I8.23` flag 2):** a negative test counts only if (i) the unmutated baseline passes in the same harness, (ii) the
  mutation is one the checker reads, (iii) the exit is the checker's failure, not a crash, (iv) the failure names the target. `w23_fold/negative_tests7.sh`
  implements all four plus a crash self-test (7/7; self-test OK).
- ⛔ **FORBIDDEN:** citing any "N/N negative tests exit 1" from this programme as evidence without its harness satisfying (i)–(iv), or at least naming
  its targets.
- **Registry:** `N400`.

### **CM-W22-4** ⭐ PoE's BUDGET AND THE PLACEHOLDERS: **27.135× (the JSON leaf; 27.14 at 2 dp; 27.13 was a truncation); three w16 per-arm placeholders PERMANENTLY BLOCKED**
- **Claim, as the artefacts support it:** PoE is 27.135× the 524,288 B reference (14,226,660 B; exact 27.13520050048828125). LACE is 31.08× (63.10×
  with whitening). Neither is byte-matched (`CM-W15-3` stands). Both drafts print 27.135× (`N399`).
- **Placeholders:** `\PLACEHOLDER{` 3 per draft, all the w16 CMAZE-T per-arm panels, **permanently blocked** (w16 persisted no arm object). The ladder's
  32,768-row **table** slot is filled; its **figure** panel (b) is not (`N401`).
- ⛔ **FORBIDDEN:** 27.13× for PoE; *"two"* w16 placeholders.
- **Registry:** `N399`, `N401`.

### **CM-W22-5** ⚠ THE w22 UNITS NO PASS HAD NAMED (draft-rulings, figures-redraw2, paper-visuals): **four new items; nothing else outstanding**
- AMENDMENT-1's (b) rendered-deviation range *"1.2e-09 … 7.8e-08"* is wrong for the item scope: the 9-leg range is 5.829167148974079e-10 …
  7.812795388417726e-08. (b) is still exact within its bar.
- The Head's AMENDMENT-2 retired the "value still answered after deletion (32.2500)" row.
- The 16,384-row comparators' 1,280 steps are derived, not stored.
- `paper-visuals/build.sh` does not rebuild in place.
- ⚠ draft-rulings' *"+0.128 is wrong"* holds only for the per-seed-best estimand; 0.12802168076341014 is the R\*-by-mean margin (`N385`).
- **Registry:** `N403`, `N404`.

### **CM-W17-10** ⚠ OPEN-ITEMS LEDGER, PASS 7 — **refit variance: DECLARED NOT RUN by the Head; the w15 fold: CLOSED; the register: extended to the W22 numbers**
| item | status after pass 7 |
|---|---|
| PRNG-replicate rival fits (`CM-W21-6`, `§I8.21` flag 1) | ⛔ **DECLARED NOT RUN**, the Head: *"not cheap enough"*. The single-fit caveat stands as printed |
| the w15 fold (`bar-d-score` / `coverage-control` / `lace-poe-rows`) | ⭐ **CLOSED** (`CM-W22-2`) |
| PoE 27.13× | ⭐ **CORRECTED** to 27.135× in both drafts (`CM-W22-4`) |
| the w16 per-arm placeholders | ⛔ **PERMANENTLY BLOCKED** (3) |
| the pre-W22 draft numbers outside any live draft cell (985 / 989) | ⚠ **OPEN**, `§I8.23` flag 3 |
| the 32,768-row ladder figure panel | ⚠ **OPEN**, figures owner (`§I8.21` flag 4 stands) |

## §I8.23 ⚠ OPEN FLAGS FOR THE HUB: the ◈ pass 7

These are raised and deliberately not resolved by the curator.

1. ⛔ **Refit variance** (`§I8.21` flag 1) is **answered by the Head's ruling**: the PRNG-replicate fits are not run (*"not cheap enough"*). ⇒ The paper
   must then carry the X1 single-fit caveat beside the single-fit margins it bears on (`N394`). Whether it does is the paper Hub's to check.
2. ⚠⚠ **The negative-test rule** (`N400`, `CM-W22-3`). Adopt criteria (i)–(iv) programme-wide? The other checkers that report "negative tests exit 1"
   (provenance, style-diff, figure gates) were not audited at this pass. ⇒ Owner: the Head.
3. ⚠ **The pre-W22 draft numbers** (985 / 989 tokens; `w13-seeds-and-tables` carries 303 tokens in `DRAFT-v2.tex`, 4 of them cells; `w16-maze-wm` 262, 14 of them cells; the rest are spread over 20+ SRC roots). Fold
   them into the register in a later seeds-and-tables pass, or declare the writers' provenance checks sufficient? ⇒ Owner: the Head / the paper Hub.
4. ⚠ **Two `% SRC` fixes and one build fix owed** (`N402`, `N403`): the `0.6407` and `16 of 16` lines; `paper-visuals/build.sh` (`REPO` as an argument).
   ⇒ Owners: the paper Hub and the figures spoke.
5. ⚠ **v2.28 is PROPOSED.** ⛔ This pass does not confirm v2.27 or below. The tiers of `N398`–`N405` are curator-proposed.
