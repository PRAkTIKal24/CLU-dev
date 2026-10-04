# LOG — pw1-draft-sweeps (plan items 2, 3, 5 + "the engine's judge")
- 2026-09-25T10:56:41Z START. Read COMMON.md, draft-sweeps.md, PLAN.md in full.
- Lock guard: live sessions pw1-plot-maze, pw1-render-shapes (no ABORT pattern). I hold the .tex lock.
- Snapshot of both live drafts + PDFs: snap-before/SHA256SUMS.
- Tools: w21_figures/tools/{style_diff_check.py,check_provenance.py,compile_check.sh} (worktree copy).
- 10:58Z baseline compile both drafts: exit 0, ucit 0, uref 0, 50 pages each; style-diff ALL EMPTY; provenance DRAFT-v2 7 unsourced/8 not-found, style 7/9. Reading the draft in full before writing sweep rules.
- 10:58Z read DRAFT-v2 L1-1898; continuing.
- 11:11Z INSTALLED both drafts + both PDFs into w17-paper (lock guard clean; live files cmp-equal to snapshot before install). Hashes: installed/SHA256-install.txt
- 11:13Z FINDINGS.md, REPLACEMENTS.md, RESIDUALS.md, word-diffs written. Choices logged: field store->field-mode n-CLU; non-n-CLU stores->memory; LACE/PoE budgets->settings; gate->control check; per-step exactness->per-step accuracy; mu^-2 budget law->scaling law; two-senses-of-exact sentence removed (no new 'exact').
- 11:13Z committed d80809f; bundle created and verified. END.
