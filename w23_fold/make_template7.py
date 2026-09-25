"""w23 fold: append §17 to the register template and write cells_pass7.py with the hand-checked extra cells.
usage: make_template7.py <gen dir (from gen_cells_pass7.py)> <out dir>"""
import json, sys, re
from pathlib import Path

G, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
rows = json.load(open(G / 'rows.json')); st = json.load(open(G / 'gen_stats.json'))
src = (G / 'cells_pass7.py').read_text(encoding='utf-8')
# Extra cells for the tokens with no verbatim line in the file their SRC names (checked by hand, w23 LOG).
EXTRA_SRC = {'S7X1': 'w21-maze-t-seed4/MAZE-T-SEED4.json', 'S7X2': 'w21-maze-t-seed4/TABLES.md', 'S7X3': 'w21-rel-newseeds/FINDINGS.md'}
EXTRA = {
    'p7_x_1': ('D5V', 2115, '0.6407'), 'p7_x_2': ('D5S', 2103, '0.6407'),
    'p7_x_3': ('S7X1', 14, '"cut": 0.6407297594090047'), 'p7_x_4': ('S7X2', 11, 'cut = 0.6407'),
    'p7_x_5': ('D5V', 2130, '16 of 16'), 'p7_x_6': ('D5S', 2118, '16 of 16'),
    'p7_x_7': ('S7X3', 62, 'random_atoms matches the CLU exactly on all 16 held-out tuples of seed 4'),
    'p7_x_8': ('D5V', 2130, 'random-atoms control and the best relational rival equal the store on 16 of 16 tuples'),
}
src = src.replace('SRC7 = {', 'SRC7 = {' + ''.join(f'"{k}": "{v}", ' for k, v in EXTRA_SRC.items()), 1)
src = src.rstrip().rstrip('}') + ''.join(f"    {k!r}: {v!r},\n" for k, v in EXTRA.items()) + '}\n'
(OUT / 'cells_pass7.py').write_text(src, encoding='utf-8')

sv, ss = st['stats']['V'], st['stats']['S']
L = ['', '## 17. ⭐ w23 FOLD (register pass 6): THE NUMBERS W22 WROTE INTO THE DRAFTS, EACH NOW A DRAFT CELL AND A SOURCE CELL', '',
     '**Why.** Pass 5 closed the w15 fold: the 199 numbers the drafts print under the three w15 spokes are cells. Every OTHER number the',
     'drafts print was checked only once, by the provenance checker of the spoke that wrote it, and by no cell of this register. A scan of the',
     'installed drafts (`w23_fold/register_coverage.py`) finds 2293 (`DRAFT-v2.tex`) and 2298 (`DRAFT-v2-style.tex`) number tokens under a',
     '`% SRC` scope, of which 212 each were draft cells here (199 w15 + 13 w21). This section folds the part W22 wrote: every number token on',
     'a line of the pinned pass-5 draft that is not in the pinned pass-4 draft (w22 draft-rulings, figures-redraw2, figures3 and seeds5 all',
     'wrote there), under a `% SRC` scope, that was not already a pass-5 draft cell on that line.', '',
     '**What each row proves, and what it does not.** Column 2 is a DRAFT cell: the token is verbatim on that line of the pinned installed',
     'draft (the live drafts are byte-identical to the pins at this pass). Column 3 is a SOURCE cell: the token, or a mechanical spelling of it',
     '(sign dropped, the minus sign swapped, comma grouping added or removed, `a of b` as `a/b` or `a / b`), is verbatim on that line of the',
     'FIRST file its `% SRC` names that exists under the evidence root, not glued to another digit. ⚠ A source cell proves PRESENCE in the',
     'cited file, the standard of the provenance checker and of the registry\'s number audit. It does not prove that the line means what the',
     'draft says. Pass 5\'s three-way check (JSON leaf at the printed precision) is NOT claimed here. Rows marked **short** carry fewer than four',
     'digits, so their presence is weak evidence; rows marked **4+** carry four or more.', '',
     f'**Counts** (`gen_stats.json`). `DRAFT-v2.tex`: {sv["w22_lines"]} W22 lines, {sv["tokens_on_w22_lines_under_SRC"]} tokens under a SRC, '
     f'{sv["already_pass5_draft_cells"]} already pass-5 cells, **{sv["registered"]} registered** ({sv["registered_strong_4plus_digits"]} of them 4+), '
     f'{sv["unsourced"]} with no verbatim source line (§17.3), {sv["tokens_on_w22_lines_without_SRC"]} with no SRC at all. `DRAFT-v2-style.tex`: '
     f'{ss["w22_lines"]} lines, {ss["tokens_on_w22_lines_under_SRC"]} tokens, {ss["already_pass5_draft_cells"]} already cells, **{ss["registered"]} '
     f'registered** ({ss["registered_strong_4plus_digits"]} 4+), {ss["unsourced"]} with no verbatim source line, {ss["tokens_on_w22_lines_without_SRC"]} with no SRC. '
     f'{st["n_source_cells"]} distinct source cells in {st["n_source_files"]} source files.', '',
     '⛔ **Not folded here, declared:** the numbers the drafts print on lines written BEFORE W22 (outside the W22 hunks). They are outside this',
     'brief. They remain checked only by their writers\' provenance checks and, where a pass-2 to pass-4 draft cell exists, against the pinned',
     'historical draft text, not the live one.', '']
for k, title, fn in (('V', '17.1 `DRAFT-v2.tex`', 'DRAFT-v2.pass5-6cecd36a.tex'), ('S', '17.2 `DRAFT-v2-style.tex`', 'DRAFT-v2-style.pass5-72327303.tex')):
    L += [f'### {title} (pinned `draft_pass5/{fn}`)', '', '| # | the draft (DRAFT cell) | the cited source (SOURCE cell) | digits |', '|---|---|---|---|']
    for i, (did, fid, strong) in enumerate(rows[k], 1):
        L.append(f'| {i} | {{{{{did}}}}} | {{{{{fid}}}}} | {"4+" if strong else "short"} |')
    L.append('')
L += ['### 17.3 ⚠ The tokens with no verbatim line in the file their SRC names (2 per draft), and what does carry them', '',
      '| draft cells | what the cited files hold | status |', '|---|---|---|',
      '| {{p7_x_1}}; {{p7_x_2}} | the line cites `MAZE-T-SEED4.json`, which holds {{p7_x_3}}; the rounded value is verbatim only in an UNCITED file, {{p7_x_4}} | ⚠ a correct 4-dp rounding of the cited leaf; the SRC should also name `w21-maze-t-seed4/TABLES.md` (draft owner) |',
      '| {{p7_x_5}}; {{p7_x_6}}; context {{p7_x_8}} | {{p7_x_7}} | ⚠ the control half is in the cited FINDINGS, worded "all 16"; the relational-rival half is NOT in it: it rests only on `w22-figures-redraw2` (fig_maze_rel_tuples: "The best +rel rival … also equals it on 16/16"), not cited on this line; the pass-6 curator re-count covers the control half only (draft owner) |',
      '']
(OUT / 'TABLES.pass7.frag.md').write_text('\n'.join(L), encoding='utf-8')
print('frag lines', len(L), 'rows', len(rows['V']), len(rows['S']))
