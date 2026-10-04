# pw1-draft-wording2 — item 7 completion: the two seed-5 numbers
Session `pw1-draft-wording2` · worktree `~/clu/worktrees/CHLU-pw1-draft-wording2` · branch `pw1/draft-wording2` (from `pw1/draft-wording` 29be0f8; its `edit.py` and the checkers are there) · out `~/clu/outputs/pw1-draft-wording2/`
**You hold the `.tex` lock.** Read `COMMON.md`, PLAN.md §1 item 7 (first bullet), and `~/clu/outputs/pw1-draft-wording/FINDINGS.md`.

## The one job
pw1-draft-wording wrote the seed-5 limitation sentence without the 0.04 margin and the 106-of-317 count, because the record was not on batmac. The Hub has now staged it, verified per file (19 files): `~/clu/evidence/outputs/w23-seed5-diagnosis/` (`FINDINGS.md` l.22–33 and l.66–78, `Q1-CONTENT*.json`, `W23-SEED5-DIAGNOSIS.json`, `LOG.md`).
- Add both numbers to that sentence in BOTH drafts, as item 7 words them. The code decodes on the wrong side of a 0.04 margin in the deployed span, and it is already wrong on 106 of the 317 fit rows that carry it.
- **0.04** is printed verbatim in the record. **106 is arithmetic**: the record prints "211 / 317 exact" and "emits S 211, E 63, N 43", so 106 = 63 + 43 = 317 − 211. Verify each of 0.04, 317, 211, 63, 43 verbatim in the record, preferring the JSON leaf over prose where one exists. Print value → file → path. Say in the `% SRC` comment that 106 is 317 − 211.
- Replace the `%` comment saying the mechanism rests on the plan's authority alone: it now rests on the record. Cite it.
- Nothing else changes. In particular, do not re-open items 8 or 13.

## Acceptance
Both drafts: exit 0, 0 undefined citations/references, style diff-check EMPTY, provenance identical or improved (the two new numbers must be found in their cited file), `\PLACEHOLDER` / `\TODO` byte-identical, page counts. Before → after of the one sentence. Installed byte-identically.
**What would prove this wrong:** any of the five numbers is not in the record, or it disagrees with the item-7 wording. Then STOP, print what the record says, and install nothing.
