"""Pass-7 build: INSERT-ONLY edits to the pass-6 registry and matrix. Never replaces or deletes a base line.
usage: python3 build_pass7.py <dir with base_pass6/{negative_results.md,claims_matrix.md}> <dir with reg_block7.md, cm_block7.md> <out dir>"""
import sys, pathlib
BASE = pathlib.Path(sys.argv[1]); S = pathlib.Path(sys.argv[2]); OUT = pathlib.Path(sys.argv[3])
reg = (BASE / 'negative_results.md').read_text().split('\n')
cm = (BASE / 'claims_matrix.md').read_text().split('\n')

def insert_before(lines, prefix, new):
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx) == 1, (prefix, len(idx))
    lines[idx[0]:idx[0]] = new + ['']

P7 = '- ⟲ **PASS 7 (2026-09-25)'
# ---------------- registry ⟲ lines (key = the NEXT heading after the annotated entry) ----------------
R = {
'### N361 —': [f'{P7}: PoE\'s MULTIPLE IS 27.135×, NOT 27.13×.** `GENTABLE-KSTAR.json` gives `x_store_reference` 27.135 on every seed; 14,226,660 /',
  '  524,288 = 27.13520050048828125, i.e. 27.14 at 2 dp. The 27.13× above (copied from w15\'s FINDINGS) is a truncation. Both drafts now print 27.135× (`N399`).',
  '  LACE\'s 31.08× and 63.10× are correct roundings. ⛔ Body not edited.'],
'### N374 —': [f'{P7}: THE 1,280 IS DERIVED, NOT STORED.** The comparator\'s 1,280 steps are 20 epochs × ⌈16,384/256⌉ batches (`figures-redraw2`',
  '  discrepancy 2); no artefact stores it as a field (`N403` item 3). The number is right; its provenance is a derivation. ⛔ Body not edited.'],
'### N395 —': [f'{P7}: THE MEASUREMENT THAT WOULD SETTLE THIS IS DECLARED NOT RUN.** The PRNG-replicate rival fits (k replicates of the carrying rival',
  '  per seed, at the carried budgets) are not run, by the Head\'s ruling: *"not cheap enough"* (`N405`). ⇒ The caveat above stands as printed and must travel',
  '  with the single-fit margins it bears on. ⛔ Body not edited. Matrix: `CM-W17-10`.'],
'## ◈.20 ': [f'{P7}: ONE MORE UNIT CLOSED AFTER THIS PASS.** `w22` seeds-and-tables pass 5 ended `2026-09-25T06:39:59Z`, after pass 6\'s END',
  '  (`2026-09-25T06:30:16Z`). It is folded at `N398`–`N402`. Three w22 units no pass had named (`draft-rulings`, `figures-redraw2`, `paper-visuals`) are',
  '  folded at `N403` (`◈.21`–`◈.23`, below).'],
}
for k, v in R.items():
    insert_before(reg, k, v)
header = ['> ◈◈◈◈◈◈◈ **WAVE-22 SEVENTH PASS — `N398`–`N405` (2026-09-25, `w23 spoke/fold`, `doc-curator`, fold symbol **◈**, pass 7).**',
 '> ⭐ **Read `◈.21` first.** A timestamp gap: seeds-and-tables pass 5 ended 06:39:59Z, after pass 6 (06:30:16Z). **The w15 fold is CLOSED** (199/199',
 '> three ways; `N398`), ⚠ scoped to the three w15 spokes. **PoE is 27.135×, not 27.13×** (`N399`). ⚠⚠⚠ **A negative test that passed by crashing',
 '> verified nothing; folded as a defect class, reproduced, and a stricter harness added** (`N400`). Three w16 placeholders permanently blocked (`N401`).',
 '> ⭐ **The numbers W22 wrote into the drafts are now register cells** (1,094 / 1,095; `N402`). The w22 sweep: `N403`. Provenance: `N404`. ⛔ The',
 '> PRNG-replicate fits are DECLARED NOT RUN by the Head (`N405`). ⟲ `PASS 7` lines at `N360`, `N373`, `N394`, `N397`. ⛔ Nothing replaced, deleted or',
 '> renumbered. Nothing written to the drafts, `figures/` or `paper-visuals/`. ⚠ All `◈` tiers are CURATOR-PROPOSED.', '']
reg[1:1] = header
tail = ['', f'{P7}: A SEVENTH PASS WAS NEEDED, BY THE RULE ABOVE.** `w22` seeds-and-tables pass 5 started `2026-09-25T06:17:43Z`, after this pass\'s START,',
        '  and ended `2026-09-25T06:39:59Z`. It is folded at `N398`–`N402`; the rest of the w22 sweep at `N403`–`N405` (`◈.21`–`◈.23`, below).']
reg_text = '\n'.join(reg).rstrip('\n') + '\n' + '\n'.join(tail) + '\n' + (S / 'reg_block7.md').read_text()
# ---------------- matrix ----------------
C = {
'### **CM-W15-4**': ['- ⟲ **PASS 7 (2026-09-25): PoE IS 27.135×, NOT 27.13×** (`CM-W22-4`, `N399`): the JSON leaf is 27.135 and 14,226,660 / 524,288 = 27.13520050048828125.',
  '  The FORBIDDEN clause above stands with the corrected figure: LACE / PoE are 31.08× and 27.135× the reference, never byte-matched.'],
'### **CM-W22-1**': ['- ⟲ **PASS 7 (2026-09-25): DECLARED NOT RUN.** The PRNG-replicate fits that would put a refit interval beside every per-corner CI are not run,',
  '  by the Head\'s ruling: *"not cheap enough"* (`N405`, `CM-W17-10`). The caveat above therefore stands as printed.'],
'2. ⛔⛔ **CMAZE-T\'s paper sentence**': ['   ⟲ **PASS 7 (2026-09-25):** answered by the Head: the PRNG-replicate fits are NOT RUN (*"not cheap enough"*). Carried to `§I8.23` flag 1.'],
'7. ⚠ **v2.27 is PROPOSED.**': ['   ⟲ **PASS 7 (2026-09-25):** flag 6 VERIFIED read-only: `\\PLACEHOLDER{` now reads **3** per draft (pass 5 filled the ladder table slot), 14',
  '   `\\figslot`s each; `fig_headline_steps` and the `rc_w19_t7_*` panels are not placed (`N401`).'],
}
for k, v in C.items():
    idx = [i for i, l in enumerate(cm) if l.startswith(k)]
    assert len(idx) == 1, (k, len(idx))
    cm[idx[0]:idx[0]] = v if not k.startswith('###') else v + ['']
title = ('# Claims-Consistency Matrix (M2/P4) — **v2.28 (the WAVE-22 SEVENTH PASS — `w23 spoke/fold`, `doc-curator`, 2026-09-25, fold symbol ◈ pass 7) — '
 '⚠ PROPOSED, PENDING HUB CONFIRMATION. ⭐⭐ WHAT THIS VERSION ADDS: `§I8.22` — `CM-W22-2` (the w15 fold CLOSED, and its scope; the W22 draft numbers '
 'made register cells), `CM-W22-3` (the negative-test defect class: a test that passes by accident verifies nothing), `CM-W22-4` (PoE 27.135×; three '
 'w16 placeholders permanently blocked), `CM-W22-5` (the w22 units no pass had named), `CM-W17-10` (open-items ledger, pass 7: PRNG-replicate fits '
 'DECLARED NOT RUN by the Head); `§I8.23` open flags; ⟲ PASS 7 lines at `CM-W15-3`, `CM-W21-6` and `§I8.21` flags 1 and 6. ⛔ This line is INSERTED '
 'above the v2.27 title, which follows VERBATIM and unedited (pass 7 replaces no line).**')
cm[0:0] = [title, '']
cm_text = '\n'.join(cm).rstrip('\n') + '\n' + (S / 'cm_block7.md').read_text()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'negative_results.md').write_text(reg_text)
(OUT / 'claims_matrix.md').write_text(cm_text)
print('ok', len(reg_text.split('\n')), len(cm_text.split('\n')))
