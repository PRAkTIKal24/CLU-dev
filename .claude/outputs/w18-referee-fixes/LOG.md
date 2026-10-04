# LOG w18 referee-fixes
- 2026-09-22T23:33:55Z start; read AMENDMENT-1, w18 COMMON, task file
- read REFEREE-v2.md, VENUE.md, full DRAFT-v2.tex, checker. Backed up both drafts to pre/ (SHA256.txt); ran checker on pre-edit pair -> pre/STYLE-DIFF-CHECK-pre.md
- M2 evidence: Table 1 rivals = w12-deletion-baselines as_trained (value-holdout pool, 16,384 rows, ~130k params); Table 3 rivals = spoke A w4 set (_thinkpad/w4/spoke-a-rivals, montero_combination pool 32,256 rows, holds_out_any_value False, 117,878 params mlp s0). Shipped leg (TABLES.md T8) scores spoke-A rivals on the 109 corners.
- 23:43Z wrote apply_fixes.py; first run on work/ copies
- 23:45Z installed edited drafts into w17-paper/; checker ALL EMPTY on installed files; compiles 35pp/36pp exit 0
- 23:47Z wrote REFEREE-FIXES.md (+copy in w17-paper/), FINDINGS.md
- 23:47Z committing branch w18/referee-fixes and bundling
- 23:47Z bundle created and verified (8013308). DONE.
