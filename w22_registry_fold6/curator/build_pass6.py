"""Pass-6 build: INSERT-ONLY edits to the pass-5 registry and matrix. Never replaces or deletes a base line.
usage: python3 build_pass6.py <curator dir with base_pass5/, reg_block6.md, cm_block6.md> <out dir>"""
import sys, pathlib
S = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
reg = (S / 'base_pass5/negative_results.md').read_text().split('\n')
cm = (S / 'base_pass5/claims_matrix.md').read_text().split('\n')

def insert_before(lines, prefix, new):
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx) == 1, (prefix, len(idx))
    lines[idx[0]:idx[0]] = new + ['']
def insert_after(lines, prefix, new):
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx) == 1, (prefix, len(idx))
    lines[idx[0] + 1:idx[0] + 1] = new

P6 = '- ⟲ **PASS 6 (2026-09-25)'
# ---------------- registry ⟲ lines (key = the NEXT heading after the annotated entry) ----------------
R = {
'### N373 —': [f'{P6}: A NEW STANDING CAVEAT ON THIS MARGIN'"'"'S PER-SEED DIGITS.** The w21 ladder X1 check shows that one 2,560-step rival fit moves by',
  '  −2.15 … +4.13 novel/corner on the PRNG draw alone, with per-corner CIs excluding 0 on 12/15 legs (`N394`). This entry\'s per-corner CIs (lowest lower',
  '  bound +3.73) do not contain refit variance. Seed 2\'s +5.02 clears the largest swing by only 0.8850152905198776 (CA6). ⭐ The sign and the 3-seed mean (+7.03) are not',
  '  threatened. ⛔ Body not edited. Matrix: `CM-W21-6`.'],
'### N374 —': [f'{P6}: INDEPENDENTLY CONFIRMED ON THE LADDER.** w21 `ladder-rivals-32k`: at matched steps, four times the rows moves every rival family',
  '  by −3.5 … +3.9 with no consistent sign, and doubling `learned_ref_ownpool`\'s steps (2,560 → 5,120) more than triples its score (`N393`). ⚠ Differences',
  '  under ≈ 4 on one seed (this entry\'s "≈ 0.5 for 8× data" included) are inside single-fit RNG variance (`N394`). ⛔ Body not edited.'],
'### N378 —': [f'{P6}: THE ADVISOR\'S MF-2 QUESTION, MEASURED.** Rivals fitted on the store\'s own 32,768 content rows (set C) give +6.08 / +11.48 /',
  '  +14.00 / +9.47 / +10.36, within −1.21 … +1.45 of this entry\'s 131,072-row set per rung, in both directions. Every CI lower bound is above 0 (`N393`).',
  '  ⚠ The ladder\'s lowest per-seed margins (+3.85 here and +3.25 step-matched, both n = 2 seed 2) sit below the largest single-fit RNG swing, 4.13',
  '  (`N394`). ⛔ Body not edited. Matrix: `CM-W21-5`.'],
'**Reading key (w21).**': [f'{P6}: ITEM 7 IS NOW ANSWERED BY MEASUREMENT, NOT BY A PREREG CITATION.** `N393`: MF-2 is real on paper and immaterial',
  '  measured (C +10.28 against B +9.97, mean over rungs; pool identity proven on 15/15 legs).'],
'### N384 —': [f'{P6}: EXTENDED TO SIX SEEDS ON TEST.** CLU-REL-A10 reads 0.925 on six seeds, at the ceiling on every seed (`N391`). ⛔ The coding',
  '  finding holds and sharpens: rivals given the relational columns leave the CLU +0.031 on average (−0.007 … +0.165; seed 2 is the only large',
  '  residual). On seed 4 TEST a random-atom store ties the CLU on all 16 tuples, and on seed 5 a matched rival ties it (`N392`). ⚠ The value row is below',
  '  the best matched rival on all six seeds. ⛔ Body not edited. Matrix: `CM-W21-4`.'],
'### N385 —': [f'{P6}: THE SAME TEST ROWS, WITH CLU-REL-A10.** On these six seeds the relational CLU reads 0.908 / 0.894 / 0.984 / 0.927 / 0.900 /',
  '  0.939 against A1\'s row above. Seed 5, A1\'s −0.185 seed, becomes a tie (0.939 = 0.939) (`N391`, `N392`). ⛔ The A1 numbers above are unchanged.'],
'### N387 —': [f'{P6}: THE TIE REPEATS ON TEST.** Seed 4 BOTH TEST: `control: random_atoms` = CLU = 0.9004149377593361, equal on 16 of 16',
  '  held-out tuples (curator count from the zero-fit replay). `landscape_removed` reads 0.2656. On seed 5 TEST the matched 80k transformer ties the',
  '  CLU at 0.939 (`N392`). ⛔ Body not edited.'],
'### N389 —': [f'{P6}: CONFIRMED ON TEST.** CLU-REL-A10 BOTH TEST reads 0.894 (seed 1) and 0.900 (seed 4), where A1 read 0.566 and 0.533 (`N391`).'],
}
for k, v in R.items():
    insert_before(reg, k, v)
header = ['> ◈◈◈◈◈◈ **WAVE-22 SIXTH PASS — `N391`–`N397` (2026-09-25, `w22 spoke/registry-fold` pass 6, `doc-curator`, fold symbol **◈**, pass 6).**',
 '> ⭐ **Read `◈.18` first.** Three units closed after pass 5 and are folded here. **`rel-newseeds`:** CLU-REL-A10 reads 0.925 on six seeds against A1\'s',
 '> 0.677, at the ceiling on every seed (`N391`). ⛔⛔ **But the margin is the input coding, not the store**: `+rel` rivals come within +0.031, a random-atom',
 '> store ties on seed 4 TEST (16/16 tuples), and seed 5 has no margin (`N392`). **`ladder-rivals-32k`:** MF-2 is immaterial when measured (C +10.28',
 '> against B +9.97; `N393`). ⚠⚠⚠ **A single 2,560-step rival fit moves by up to 4.13 novel/corner on the PRNG draw alone, and per-corner CIs do not',
 '> contain that (`N394`).** **`figures3`:** the CMAZE-T panels hid facts the judge scores (33,412 of 92,628 pairs); now 0, and no number was affected',
 '> (`N395`). Provenance: `N396`. Omit-nothing sweep: `N397`. ⟲ `PASS 6` lines at `N372`, `N373`, `N377`, `N381`, `N383`, `N384`, `N386`, `N388` and `◈.17`.',
 '> ⛔ Nothing replaced, deleted or renumbered. ⚠ All `◈` tiers are CURATOR-PROPOSED.', '']
reg[1:1] = header
tail = ['', f'{P6}: THE LINE ABOVE STILL HOLDS FOR THOSE UNITS.** Three other units closed after this pass\'s END: `rel-newseeds` (`2026-09-24T16:16:03Z`),',
        '  `ladder-rivals-32k` (`2026-09-24T22:03:55Z`) and `figures3` (`2026-09-24T22:13:46Z`). They are folded at `N391`–`N397` (`◈.18`–`◈.20`, below).']
reg_text = '\n'.join(reg).rstrip('\n') + '\n' + '\n'.join(tail) + '\n' + (S / 'reg_block6.md').read_text()
# ---------------- matrix ----------------
C = {
'### **CM-W20-2**': ['- ⟲ **PASS 6 (2026-09-25): SINGLE-FIT RNG CAVEAT.** One 2,560-step rival fit moves by up to 4.13 novel/corner on the PRNG draw alone',
  '  (`CM-W21-6`). ⛔ Add to FORBIDDEN: per-seed digits of +7.03 (esp. seed 2\'s +5.02) presented as precise on per-corner CIs alone. ⭐ The sign and the',
  '  3-seed mean stand.'],
'### **CM-W20-3**': ['- ⟲ **PASS 6 (2026-09-25): CONFIRMED ON THE LADDER** (`CM-W21-5`): at matched steps, 4× the rows moves every family −3.5 … +3.9, mixed sign.'],
'### **CM-W21-1**': ['- ⟲ **PASS 6 (2026-09-25): MF-2 MEASURED** (`CM-W21-5`). Set C (the store\'s own 32,768 rows) gives +6.08 / +11.48 / +14.00 / +9.47 /',
  '  +10.36, within −1.21 … +1.45 of the 131k set. The approved ladder wording above stands; name pool **and** steps.'],
'### **CM-W21-3**': ['- ⟲ **PASS 6 (2026-09-25): SIX SEEDS** (`CM-W21-4`). 0.925 against A1\'s 0.677; `+rel` residual +0.031 on average; seed 4 TEST',
  '  `random_atoms` = CLU (16/16 tuples); seed 5 tied. Statement 2 above generalises: the margin is the coding. The forbidden list above stands.'],
'## §I8.19 ': ['- ⟲ **PASS 6 (2026-09-25):** MF-2 is now **measured** (`CM-W21-5`); single-fit RNG variance is a new standing caveat (`CM-W21-6`); see `CM-W17-9`.'],
}
for k, v in C.items():
    insert_before(cm, k, v)
insert_after(cm, "8. ⚠ **CLU-REL-A10's store gate**", ['   ⟲ **PASS 6 (2026-09-25):** the seed-4 gate failure REPEATS on TEST (0.9004, 16/16 tuples; `N392`). Carried to `§I8.21` flag 3.'])
title = ('# Claims-Consistency Matrix (M2/P4) — **v2.27 (the WAVE-22 SIXTH PASS — `w22 spoke/registry-fold` pass 6, `doc-curator`, 2026-09-25, fold symbol ◈ pass 6) — '
 '⚠ PROPOSED, PENDING HUB CONFIRMATION. ⭐⭐ WHAT THIS VERSION ADDS: `§I8.20` — `CM-W21-4` (CLU-REL-A10 on six seeds, 0.925 vs 0.677: the margin is the '
 'input coding, not the store), `CM-W21-5` (MF-2 measured: immaterial; steps, not pool), `CM-W21-6` (single-fit RNG-draw variance up to 4.13 novel/corner: '
 'a standing caveat on every single-fit margin), `CM-W22-1` (the CMAZE-T figure defect, measured and fixed, no number affected), `CM-W17-9` (open-items '
 'ledger, pass 6); `§I8.21` open flags; ⟲ PASS 6 lines at `CM-W20-1`, `CM-W20-2`, `CM-W20-4`, `CM-W21-2`, `CM-W17-8` and `§I8.19` flag 8. ⛔ This line '
 'is INSERTED above the v2.26 title, which follows VERBATIM and unedited (pass 6 replaces no line).**')
cm[0:0] = [title, '']
cm_text = '\n'.join(cm).rstrip('\n') + '\n' + (S / 'cm_block6.md').read_text()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'negative_results.md').write_text(reg_text)
(OUT / 'claims_matrix.md').write_text(cm_text)
print('ok', len(reg_text.split('\n')), len(cm_text.split('\n')))
