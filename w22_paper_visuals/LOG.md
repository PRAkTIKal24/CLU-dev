# LOG — w22 spoke/paper-visuals (batmac)

- 2026-09-24T17:55:58Z START. Brief read; paper-visuals README/FIGURES/PROVENANCE read.
- Placement: both drafts include figures only via \figslot{stem}{caption}{label} (macro -> \includegraphics{figures/stem.pdf}); 17 figslot calls per draft, parsed from source.
- Rebuild check: run the folder's own build.sh with BOTH destination args pointed at fresh scratch dirs
  (/Users/batmac/clu/outputs/w22-paper-visuals/rebuild-20260924T175626Z/destA, destB) so nothing installed is overwritten; then compare scratch bytes to installed bytes.
- 2026-09-24T17:57:17Z REBUILD ATTEMPT 1 (in-folder paper-visuals/build.sh, scratch dests): FAILED, EXIT 1, 0 figures produced.
  cmazet_panels.py sets REPO = parent of its own folder (= /Users/batmac/clu/outputs when run in paper-visuals)
  and asserts chlu is imported from there; outputs/ has no chlu -> AssertionError. The folder's build.sh is
  therefore NOT runnable in place. Log: rebuild-20260924T175626Z/build.log. Nothing installed was touched.
- The build code is committed on this branch at w21_figures/; all 7 scripts + 5 data JSONs cmp-identical to the
  paper-visuals copies. REBUILD ATTEMPT 2: w21_figures/build.sh from this worktree (REPO = worktree), scratch dests /Users/batmac/clu/outputs/w22-paper-visuals/rebuild2-20260924T175717Z.
- 2026-09-24T17:57:50Z REBUILD ATTEMPT 2: EXIT 0, 'REBUILD: BYTE-IDENTICAL' (build a == build b); 27/27 manifest files (26 fig + PROVENANCE.json) byte-identical to the INSTALLED paper-visuals files and to SHA256-w22-build1.txt. Rebuilt manifest == installed manifest. Nothing overwritten.
- 2026-09-24T17:59:24Z Placement from COMPILED .aux (scratch copy of w17-paper, pdflatex x2, drafts untouched): 17 fig labels per draft, Figure 1-17, identical label set in both drafts.
- 2026-09-24T17:59:24Z SUFFICIENCY TEST: scratch copy with figures/ REPLACED by a copy of paper-visuals/: both drafts compile, exit 0, 17/17 distinct figures/*.pdf included per draft, 0 "Figure pending" fallbacks, 0 LaTeX errors.
- Mirrored (non-build) files: 10 files (fig_task_* x4, rendered/rc_w19_t7_* x6) checked against their OWN build manifests
  (w19-task-figures/proof/SHA256-build{1,2}.txt; w17-paper/figures/rendered/SHA256-w19-build{1,2}.txt). Not rebuilt: the rendered
  panels need the frozen decoder forward pass (= compute, excluded by brief); task figures need 3dshapes.h5 + a venv. DECLARED NOT RUN.
- Brief claims verified from PROVENANCE cells: MLP @20,480 steps means 38.60/39.07/39.15 on 16,384/32,768/131,072 rows (spread 0.55);
  maze_conditions marked = A1 s1 rival>CLU, A1 s5 rival>CLU, REL s4 control=CLU (random atoms 0.9004), REL s5 rival=CLU (0.9394); deletion panel (d) has "1 % items (control)" beside red∧sphere.
- 2026-09-24T18:02:42Z README generated (make_index.py EXIT 0: 0 missing, 0 parity failures, 0 hash problems). FINDINGS.md written.
