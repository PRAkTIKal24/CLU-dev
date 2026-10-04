# pw1-visuals-tooling — plan item 4: `paper-visuals/` rebuilds itself in place
Session `pw1-visuals-tooling` · worktree `~/clu/worktrees/CHLU-pw1-visuals-tooling` · branch `pw1/visuals-tooling` (from `pw1/draft-figures`) · out `~/clu/outputs/pw1-visuals-tooling/`
Read `COMMON.md`, then PLAN.md §1 item 4, then `~/clu/outputs/pw1-draft-figures/FINDINGS.md`. ⛔ **You do NOT edit either draft** (a draft spoke may be running in parallel). You may write only `paper-visuals/` scripts and README, plus the matching script copies in `w17-paper/figures/`. ⛔ Zero content change: no figure byte may change.

## The one job
Fix the repo-root derivation in `cmazet_panels.py` / `build.sh` (and anything else that assumes the folder's parent is a CHLU repo) so that `bash build.sh`, run IN PLACE in `~/clu/outputs/paper-visuals/`, rebuilds every figure the folder owns byte-identical to the installed files. The Head edits plots through this loop. Make it work from a fresh copy of the folder, too. For the mirrored figures (task figures, rendered image panels), write one documented rebuild line each, naming the dataset and frozen-decoder env. The README states the one command at the top.
- Test it in a COPY first (a timestamped scratch copy of `paper-visuals/`, building into scratch destinations). Then commit the fix to your branch and install the fixed scripts into both folders. Then run `bash build.sh` in place ONCE, with destinations set to scratch dirs so no installed file is overwritten, and show that every output equals the installed file.

## Acceptance
- The in-place build exits 0. Every built file's sha256 equals the installed file's (print the table). Two builds are byte-identical.
- A fresh-copy build (the folder copied to `/tmp/…`) also exits 0 and produces identical hashes, or prints exactly which external input it needs and where the README documents it.
- Diff of script changes (code only), plus the README top section.
**What would prove this wrong:** any figure byte changes, or the build still fails when the folder is not under a repo.

## Hub notes (binding)
- Base `pw1/draft-figures` a01bea7. The installed figures were built at f571b2b (see `pw1-draft-figures/FINDINGS.md` and INSTALL-TABLE.md). Your in-place build must reproduce those installed hashes.
- Also stop `build.sh` producing `fig_headline_steps`, which has no slot and was retired to `retired-pw1/`: remove its dispatch in `make_figures_w22.py` and keep the function. This changes no placed figure. Show that the 13 placed stems are byte-unchanged.
- A draft spoke (`pw1-draft-wording`) runs in parallel. It edits only the two `.tex`. Do not write the drafts or the PDFs.
