# w23_fold — w23 spoke/fold (registry pass 7; w18 register pass 6)
- `outputs/` — a copy of `~/clu/outputs/w23-fold/` (FINDINGS, LOG, START/END, the installed registry + matrix, `curator/`).
- `register_w18_seeds_and_tables/` — the register files this pass changed or added (`cells_pass7.py`, `build_tables.py`, `TABLES.template.md`,
  `TABLES.md`, `TABLES-w18.json`, `TABLES.pass7.frag.md`, `FINDINGS.md`, `LOG.md`). ⚠ Not standalone: `build_tables.py` also imports
  `cells_pass2..5.py`, `jnum.py` and the pinned drafts from the full register (`~/clu/outputs/w18-seeds-and-tables/`, branch `w22/seeds5`).
- Tools: `gen_cells_pass7.py`, `make_template7.py`, `stage_register7.py`, `negative_tests7.sh` (+ `mutate.py` from pass 5), `register_coverage.py`,
  `curator_arithmetic_p7.py`, `build_pass7.py`, `verify_pass7.py`, `audit_pass7.py`, `rounding_check_p7.py`, `verify_insert_only.py`;
  blocks `reg_block7.md`, `cm_block7.md`. `pass5_negative_tests.sh.ref` is pass 5's suite, for reference.
