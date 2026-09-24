"""Pass-5 build: INSERT-ONLY edits to the pass-4 registry and matrix. Never replaces or deletes a base line."""
import sys, pathlib
S = pathlib.Path(sys.argv[1])          # scratchpad with base/, reg_block.md, cm_block.md
OUT = pathlib.Path(sys.argv[2])        # output dir
reg = (S / 'base/negative_results.md').read_text().split('\n')
cm = (S / 'base/claims_matrix.md').read_text().split('\n')

def insert_before(lines, prefix, new, nth=0):
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx) >= 1, prefix
    i = idx[nth]
    lines[i:i] = new + ['']
def insert_after(lines, prefix, new):
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx) == 1, (prefix, len(idx))
    i = idx[0] + 1
    lines[i:i] = new

P5 = '⟲ **PASS 5 (2026-09-24)'
# ---------------- registry ⟲ lines ----------------
R = {
'### N340 —': [f'- {P5}: MEASURED. w20 `shapes-data-matched` is the data-matched test this entry asked for.** The data-matched margin is **+7.03**',
  '  (seeds +7.49 / +8.60 / +5.02; +6.54 against the best of every fit), not +39…+49, and it does **not** collapse (`N372`). ⭐⭐ **The cause is not the',
  '  one this entry names:** at matched steps an MLP on the same 16,384 rows scores 38.6, against 5.5 at the comparator\'s 1,280 steps ⇒ the operative',
  '  asymmetry was **training schedule**, not data (`N373`). So instance 1 collapses onto instance 2\'s axis (steps). A fourth axis, a width forced by the',
  '  float match that the rival cannot train at, is measured on CMAZE-T (`N387`). ⛔ The body above is not edited. ⛔ M1 is **measured, not ruled**: the',
  '  paper\'s wording stays with the Head and the Advisor. Matrix: `CM-W20-1`, `CM-W20-2`, `CM-W17-8`.'],
'### N341 —': [f'- {P5}: M3 MEASURED UNDER THE HEAD\'S DEFINITION** (AMENDMENT-2: downdate the store **and** refit the address book from downdated',
  '  statistics). The mechanism is exact ((a) max 5.383044481277466e-07; (b1) 0), and the outcome is not: deleting `red ∧ sphere` costs the surviving',
  '  corners −6.13 / −6.43 / −6.30 on 3/3 seeds, while a random 1 % moves −0.02 / −0.11 / −0.01 (`N374`). The book\'s exact statistic is the fit data',
  '  itself (0.97 GB; `N375`). ⇒ *"the address book is carried, not refit"* is now a **measured** difference, not only a wording defect: the carried book is',
  '  what kept the w15 survivors "unchanged". ⛔ Body not edited; M3\'s wording stays with the Head and the Advisor. Matrix: `CM-W20-3`, `CM-W17-8`.'],
'### N351 —': [f'- {P5}: REALISED AT K = 32 ON SHAPES.** w20 found that the historical T7 across-wells permutation at seed 2 had **2 fixed points**; it',
  '  still read 0.000, as does the asserted derangement w20 printed on every row (`N379`; `w20-shapes-data-matched/FINDINGS.md` l.106–107). This entry\'s rule held.'],
'### N356 —': [f'- {P5}: THE SURVIVOR SIDE, UNDER A REFIT BOOK.** The rows above were measured with the address book **carried**. With the book refit',
  '  (Head AMENDMENT-2), deleting `red ∧ sphere` removes red composition with the other shapes: −32.1 per corner on the 20 red non-sphere corners (seed 0),',
  '  −6.13 / −6.43 / −6.30 over all 109 corners (`N374`). ⛔ The deleted corners\' own row (*"c still composes after forgetting c"*) is **RETIRED** from',
  '  tables by AMENDMENT-2; its evidence stays here and in w15\'s artefacts. ⛔ Body not edited.'],
'### N361 —': [f'- {P5}: RE-FIT ON THE STORE\'S 131,072-ROW POOL, THE SAME.** LACE 0.018 / 0.046 / 0.073 and PoE 0.000 on every seed, rows 3–4 (`N378`).',
  '  The standing scope above carries over verbatim. LACE/PoE @131,072 on the rungs is DECLARED NOT-RUN (cost), so this entry\'s rung rows remain the only',
  '  LACE/PoE rung rows.'],
'### N367 —': [f'- {P5}: THE VERDICT COLUMN ABOVE IS THE w19 PREREG\'S, AND IT IS SUPERSEDED FOR THE PAPER.** Under the Head\'s label rule',
  '  (`w19-advisor-report.md` §0a), BOTH TEST reads **WIN** (ahead 3/3, Δ̄ +0.0988, σ 0.0751) and OBS reads **COMPARABLE**; WORDS-MAIN and WORDS-FULL stay',
  '  LOSS (`N389`; every σ re-derived by the curator). ⚠ The per-seed-best caveat travels with the WIN (seed 1: 0.5674 against 0.5660). ⭐ What does **not**',
  '  change: the w16 A1 row stays withdrawn **as a margin credited to the store**. WORDS-FULL is LOSS on TEST too (`N382`), and the w21 improvement\'s margin',
  '  is mostly the coding (`N383`). The six-seed extension is at `N384` / `N385`. ⛔ Body not edited.'],
'### N368 —': [f'- {P5}: CONFIRMED ON TEST.** WORDS-FULL TEST (w21 runs 4–6/9): Δ̄ −0.017517974813034298, 0/3 seeds ahead, LOSS; TEST repeats VAL',
  '  (−0.0173), with the instrument reproduced bit-for-bit (`N382`). ⚠ Under the CLU\'s own or R\\*\'s per-seed spread it would read COMPARABLE. ⭐ The pattern',
  '  repeats one level up: a rival given CLU-REL-A10\'s relational columns matches the CLU on WORDS-FULL VAL (0.9414055233621977 against',
  '  0.9371139657869874; `N383`).'],
'### N369 —': [f'- {P5}: RELABELLED COMPARABLE UNDER THE HEAD\'S RULE** (σ 0.1068 > |Δ̄| 0.1053; `N389`). ⛔ The failing link is unchanged: READ. At six',
  '  seeds the CLU is behind the best matched rival on OBS VAL on **all 6** (`N384`). ⛔ Body not edited.'],
'### N370 —': [f'- {P5}: "SPENT ON A LOSS" IS THE w19 PREREG\'S VERDICT; UNDER THE HEAD\'S RULE BOTH TEST IS A WIN** (`N389`). Runs 4–6/9 were later',
  '  spent on WORDS-FULL TEST (LOSS, `N382`) and 7–9/9 on CLU-REL-A10 BOTH TEST (WIN, `N383`), so target `maze-t-symmetric` is 9 of 9. ⚠ The 4-unit',
  '  hidden layer this entry cautions about is now a **measured** defect (`N387`). ⛔ Body not edited.'],
'## ◈.14 ': [f'- {P5}: ITEM 2 IS CORRECTED IN THE HUB\'S w19 REPORT** (`w19-advisor-report.md` l.8, citing this entry). w19\'s own FINDINGS l.36–38 still',
  '  carries the unscoped sentence.'],
'## ⛔ DECLARED NOT-RUNs — the ◈ pass 4': [f'{P5}: FOLDED.** w20 `shapes-data-matched` closed (END `2026-09-24T13:36:07Z`) and is folded at `N372`–`N381`; the four',
  'w21 units at `N382`–`N388`; the w19 relabel at `N389`; the omit-nothing sweep at `N390` (`◈.15`–`◈.17`, at the end of this file).'],
}
# N339's ⟲ goes before N340's heading etc. (each key is the NEXT heading after the annotated entry)
for k, v in R.items():
    insert_before(reg, k, v)
header = ['> ◈◈◈◈◈ **WAVE-21 FIFTH PASS — `N372`–`N390` (2026-09-24, `w21 spoke/registry-fold`, `doc-curator`, fold symbol **◈**, pass 5).**',
 '> ⭐ **Read `◈.15` first.** ⛔⛔ The Head banned pre-registrations on 2026-09-24: everything withheld for want of one is folded with its status (`N380`, `N390`).',
 '> **w20 `shapes-data-matched`** (3 of 3 spent): the data-matched Shapes margin is **+7.03, not +39…+49**, and does not collapse (`N372`); ⭐⭐ **M1\'s cause was',
 '> SCHEDULE, not data** (`N373`); ⭐⭐ **M3 measured**: under a refit book, deleting `red ∧ sphere` costs survivors −6.13 / −6.43 / −6.30 (`N374`); the book\'s',
 '> exact statistic is the fit data (`N375`); `learned_1x` was not printed and is folded with its number (`N380`). **w21:** WORDS-FULL TEST **LOSS** (`N382`);',
 '> CLU-REL-A10 **WIN**, mostly the coding, zero on WORDS-FULL VAL (`N383`); six seeds: seed 5 loses by 0.185 (`N384`), and "+0.046 vs +0.099" mixes estimands',
 '> (`N385`); CLU-REL-A10\'s store gate fails on seed 4 VAL (`N386`); the matched-float rule is a measured defect (`N387`). The w19 relabel (BOTH TEST → WIN)',
 '> is `N389`. ⟲ `PASS 5` lines at `N339`, `N340`, `N350`, `N355`, `N360`, `N366`, `N367`, `N368`, `N369`, `N371` and `◈.14`. ⛔ Nothing replaced, deleted or',
 '> renumbered. ⚠ All `◈` tiers are CURATOR-PROPOSED.', '']
reg[1:1] = header
reg_text = '\n'.join(reg).rstrip('\n') + '\n' + (S / 'reg_block.md').read_text()
# ---------------- matrix ----------------
C = {
'### **CM-W17-2**': ['- ⟲ **PASS 5 (2026-09-24): MEASURED.** Data-matched margin **+7.03** (`CM-W20-1`); the operative asymmetry was **optimiser steps**, not data',
  '  (`CM-W20-2`). ⇒ The ≈ +39…+49 margin is not a data-matched number. ⛔ Wording still the Advisor\'s; see `CM-W17-8`.'],
'### **CM-W17-3**': ['- ⟲ **PASS 5 (2026-09-24): MEASURED under the Head\'s refit definition** (`CM-W20-3`): exact mechanism, but `red ∧ sphere` costs survivors',
  '  −6.13 / −6.43 / −6.30. ⛔ Wording still the Advisor\'s; see `CM-W17-8`.'],
'### **CM-W15-4**': ['- ⟲ **PASS 5 (2026-09-24):** re-fit on the store\'s 131,072-row pool, LACE / PoE score no better (0.018 / 0.046 / 0.073; 0.000). The scope',
  '  above stands verbatim. See `CM-W20-4`.'],
'### **CM-W19-4**': ['- ⟲ **PASS 5 (2026-09-24): THE VERDICT COLUMN ABOVE IS THE w19 PREREG\'S.** Under the Head\'s rule BOTH TEST is **WIN** and OBS **COMPARABLE**',
  '  (`CM-W19-5`). The forbidden-wordings list above stands; WORDS-FULL is LOSS on TEST too (`CM-W21-1`).'],
'## §I8.17 ': ['- ⟲ **PASS 5 (2026-09-24):** M1 and M3 are now **measured** (`CM-W17-8`). Wording still held for the Head and the Advisor.'],
}
for k, v in C.items():
    insert_before(cm, k, v)
insert_after(cm, '8. ⛔ **Fifth pass required** for w20', ['   ⟲ **PASS 5 (2026-09-24): DONE.** w20 is folded at `CM-W20-1`…`CM-W20-4` (`§I8.18`).'])
title = ('# Claims-Consistency Matrix (M2/P4) — **v2.26 (the WAVE-21 FIFTH PASS — `w21 spoke/registry-fold`, `doc-curator`, 2026-09-24, fold symbol ◈ pass 5) — '
 '⚠ PROPOSED, PENDING HUB CONFIRMATION. ⭐⭐ WHAT THIS VERSION ADDS: `§I8.18` — `CM-W20-1` (the data-matched Shapes margin is +7.03, not +39…+49; shrinks ~6×, '
 'no collapse), `CM-W20-2` (M1\'s mechanism: SCHEDULE, not data), `CM-W20-3` (M3 measured: exact mechanism, −6.13…−6.43 on survivors; the book\'s statistic is '
 'the fit data), `CM-W20-4` (ladder, LACE/PoE at 131k, K = 5 n = 4/5 not claimable), `CM-W21-1` (WORDS-FULL TEST LOSS), `CM-W21-2` (CLU-REL-A10 WIN, mostly the '
 'coding), `CM-W21-3` (six seeds: heterogeneous, estimand-dependent), `CM-W19-5` (the w19 relabel: BOTH TEST → WIN), `CM-W17-8` (M1 and M3 measured, wording held); '
 '`§I8.19` open flags; ⟲ PASS 5 lines at `CM-W17-1`, `CM-W17-2`, `CM-W15-3`, `CM-W19-3`, `CM-W17-7` and `§I8.17`. ⛔ This line is INSERTED above the v2.25 '
 'title, which follows VERBATIM and unedited (pass 5 replaces no line).**')
cm[0:0] = [title, '']
cm_text = '\n'.join(cm).rstrip('\n') + '\n' + (S / 'cm_block.md').read_text()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'negative_results.md').write_text(reg_text)
(OUT / 'claims_matrix.md').write_text(cm_text)
print('ok', len(reg_text.split('\n')), len(cm_text.split('\n')))
