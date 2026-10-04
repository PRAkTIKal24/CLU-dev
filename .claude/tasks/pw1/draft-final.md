# pw1-draft-final — the closing audit: final compile, figure/table index, page counts, residuals
Session `pw1-draft-final` · worktree `~/clu/worktrees/CHLU-pw1-draft-final` · branch `pw1/draft-final` (from `base/pw1-final` = pw1/draft-wording2 + --no-ff pw1/visuals-tooling; every PW-1 checker is in the tree) · out `~/clu/outputs/pw1-draft-final/`
**You hold the `.tex` lock.** Read `COMMON.md`, PLAN.md §1 in full, and every PW-1 spoke's FINDINGS: `~/clu/outputs/pw1-*/FINDINGS.md`. **You make no content edits.** The only draft edit allowed is a mechanical defect fix (a broken `\ref`, a stale comment line that states a wrong count). Print each one, and install it by the standard method.

## The one job
1. A final full compile of both drafts, from a scratch copy AND in place: exit 0, 0 undefined citations/references, 0 overfull boxes (list any), style diff-check EMPTY. The scratch and in-place PDFs are page-count equal.
2. **Figure/table index for the Head:** number → label → first line of caption → page, for every figure and table in DRAFT-v2, and the style draft's numbers/pages beside it.
3. **Page counts per section** (main text and each appendix), measured with `\typeout{\thepage}` markers on a scratch copy, for both drafts. Also main-text total, total pages, and main text against ICLR's 9 pp.
4. **Every remaining `\TODO` and `\PLACEHOLDER`**, by argument, with line and section, for both drafts. `grep -c` counts lines, not placeholders, so count arguments.
5. **Whole-wave residual audit** of rendered text in both drafts, for every rule of items 2, 3, 5, 8, 13, 14, 15, 17 and the pendings: banned words, the § / `Figure~` / `Table~` / `Appendix~` / `Section~` ref forms, hard/easy, deletion-sense "exact", CMAZE-1/2/2b/O, `corner`, `oblong`, `ledger`, `rival`, `store`, `10x`. Each residual is 0 or listed with its reason.
6. **Figure-folder audit:** every placed stem present in both `w17-paper/figures/` and `paper-visuals/` with equal hashes, and the README index matching.

## Deliverable
`FINAL-INDEX.md` in your out dir, holding sections 1–6 as tables, plus FINDINGS and "What did NOT run".
**What would prove this wrong:** an index row that disagrees with the compiled `.aux`, or a residual that the audit misses. Positive-control each scan by planting the token in a scratch copy and showing that the scan catches it.

## Hub notes (binding)
- **Step 0, before the audit: refresh the figure folders' bookkeeping.** pw1-visuals-tooling left the installed `PROVENANCE.json` (and the `SHA256-w22-build{1,2}.txt` manifests) one build behind, deliberately (see its FINDINGS "Design" 4–5). Run the real in-place build ONCE: `cd ~/clu/outputs/paper-visuals && bash build.sh` with its default destinations (both folders). Before and after, hash every placed figure file in both folders: all 26 per folder must be byte-UNCHANGED. The new `PROVENANCE.json` must equal the one visuals-tooling predicted (its three edits). Then regenerate the README with `make_index.py` in both folders. If any placed figure byte changes, STOP, restore from the pre-run snapshot, and report.
- The drafts now hold one derived number (106 = 317 − 211; raw provenance +1 not-found, verified by `pw1_draft_wording2/check_derived.py`). Report it; do not "fix" it.
- Record for the report: the title is now "naive-CLU: Deletable Compositional Generation on a Frozen Encoder".
