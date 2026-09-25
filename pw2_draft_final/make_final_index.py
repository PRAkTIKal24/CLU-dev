"""Assemble FINAL-INDEX.md (the closing audit of PW-2) from the artefacts under G. usage: python3 make_final_index.py G OUTDIR"""
import json, os, re, sys
from collections import Counter, defaultdict
G, OUT = sys.argv[1:3]
DR = [('v2', 'DRAFT-v2'), ('style', 'DRAFT-v2-style')]
rd = lambda p: open(os.path.join(G, p)).read()
L = []
P = L.append


def compile_part():
    P('## 1. Full compile of both live folders, from a clean copy')
    P('')
    P('Built from a copy of `~/clu/pw2/live/{v2,style}` without any build products (`.aux .log .bbl .blg .out .pdf` excluded), '
      'FROM the folder: pdflatex, bibtex, pdflatex until no "Rerun" (`pw2_draft_sections/compile_folder.sh`). The copied `.tex` files '
      'are cmp-equal to the live ones. The extracted text of the fresh builds equals, page by page, the text of the PDFs installed '
      'in the live folders (`checks/LIVE-PDF-VS-BUILD.txt`).')
    P('')
    P('| draft | exit codes | undefined citations | undefined references | errors | multiply defined | overfull | underfull | bibitems | Rerun left | pages |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    for f, d in DR:
        log = rd(f'{f}/{d}.log'); cl = rd(f'{f}/compile-{d}.log').strip().split('\n')[-1]
        pages = re.search(r'Output written on .*?\((\d+) pages', log).group(1)
        P('| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d | %s |' % (d, cl.replace('EXITS ', ''), len(re.findall(r'Citation .* undefined', log)),
          len(re.findall(r'Reference .* undefined', log)), len(re.findall(r'^!', log, re.M)), len(re.findall('multiply defined', log)),
          len(re.findall(r'^Overfull', log, re.M)), len(re.findall(r'^Underfull', log, re.M)), rd(f'{f}/{d}.bbl').count('\\bibitem'),
          len(re.findall('Rerun to get', log)), pages))
    P('')
    P('**Overfull boxes: none in either draft.** Underfull boxes, every one (the log line and the source lines it names; "lines 66--73" '
      'are in the `.bbl` (the Efron 1979 entry), every other line range is in the `.tex`; lines 295--304 = Methods "The read", 428/426--438/436 = '
      'Experiments "Baselines", the rest are appendix table cells):')
    P('')
    for f, d in DR:
        log = rd(f'{f}/{d}.log').split('\n')
        u = [l.strip() for l in log if l.startswith('Underfull')]
        c = Counter(re.sub(r'\(badness \d+\) ', '', x) for x in u)
        P('- %s (%d):' % (d, len(u)))
        for k, v in sorted(c.items(), key=lambda x: -x[1]):
            P('  - %s%s' % (k.replace('\\', '\\\\') if False else '`%s`' % k, ' ×%d' % v if v > 1 else ''))
    P('')
    P('BibTeX warnings, verbatim (`.blg`), both drafts identical:')
    P('')
    for f, d in DR:
        blg = rd(f'{f}/{d}.blg')
        w = [l for l in blg.split('\n') if l.startswith('Warning--') or l.startswith('(There')]
        P('- %s: %s' % (d, ' · '.join('`%s`' % x for x in w)))
    P('')
    P('Other LaTeX / package warnings (counts by kind, both drafts):')
    P('')
    for f, d in DR:
        log = rd(f'{f}/{d}.log')
        c = Counter(re.sub(r'\s+on input line \d+\.?', '', re.sub(r"`[^']*'", "`…'", m.group(0))).strip() for m in re.finditer(r'(LaTeX|Package \w+) Warning: [^\n]*', log))
        P('- %s: %s' % (d, '; '.join('%s ×%d' % (k, v) for k, v in c.most_common())))
    P('')
    P('**Mechanical defects** (`scans.py defects`): 0 rendered `??`, 0 undefined-citation marks, 0 absolute machine paths in either `.tex` '
      '(code and comments), the title renders on 2 lines in both drafts, 0 doubled words in the source, 0 forced `\\\\` breaks in prose. '
      'The rendered-text scan does list doubled words, all of them inside figure panels or adjacent table cells (text-extraction joins, '
      'not source text; `checks/DEFECTS-*.md`). **No defect to fix, so no edit was made and nothing was installed.**')
    P('')


def style_part():
    t = rd('checks/STYLE-DIFF-CHECK.md')
    P('## 2. Style diff-check between the drafts (`pw1_draft_sweeps/style_diff_check.py`, unchanged; exit 0)')
    P('')
    m = re.search(r'numbers: original (\d+) tokens \((\d+) distinct\); style (\d+) tokens \((\d+) distinct\)', t)
    ms = re.search(r'multiset diff\*\* \(every occurrence\) lost: `([^`]*)`; added: `([^`]*)`', t)
    ph = re.search(r'original: (\d+) placeholders; style: (\d+) placeholders', t)
    ap = re.search(r'artefact paths named in [^:]*: original (\d+) tokens \((\d+) distinct\); style (\d+) tokens \((\d+) distinct\)', t)
    hd = re.search(r'headings \([^)]*\): original (\d+), style (\d+); identical sequence: \*\*(\w+)\*\*', t)
    lb = re.search(r'label\{\}` sequence[^:]*: original (\d+), style (\d+); identical sequence: \*\*(\w+)\*\*', t)
    P('| check | DRAFT-v2 | DRAFT-v2-style | diff |')
    P('|---|---|---|---|')
    P('| number multiset (every digit run, whole file, code only) | %s tokens (%s distinct) | %s tokens (%s distinct) | lost %s, added %s: **EMPTY** |' % (m.group(1), m.group(2), m.group(3), m.group(4), ms.group(1), ms.group(2)))
    P('| `\\PLACEHOLDER` arguments, in order | %s | %s | **EMPTY** |' % ph.groups())
    P('| artefact paths in `%% SRC` comments | %s (%s distinct) | %s (%s distinct) | **EMPTY** |' % ap.groups())
    P('| heading sequence (section … paragraph) | %s | %s | identical: **%s** |' % hd.groups())
    P('| `\\label` sequence | %s | %s | identical: **%s** |' % lb.groups())
    P('')
    P('All four diffs EMPTY (`checks/STYLE-DIFF-CHECK.md`). The main-text digit tokens (this audit\'s scanner, §6) are also equal between the drafts, '
      'except one `3` inside the name "Shapes3D" in the style twin\'s own Head-held Experiments sentence ("This section measures the three properties on Shapes3D.").')
    P('')


def prov_part():
    P('## 3. Provenance findings against the PW-1 final state')
    P('')
    P('`pw1_draft_sweeps/check_provenance.py` (unchanged) run now on the live drafts and on the PW-1 final drafts '
      '(`outputs/pw1-draft-citations/stage-145204/post-install/`, sha256 = `SHA256-installed.txt` there = `outputs/w17-paper/`). '
      'Findings compared as multisets with line numbers masked (`prov_compare.py`).')
    P('')
    P(re.sub(r'(?m)^## (DRAFT[-\w]*)', r'#### \1', rd('checks/PROVENANCE-VS-PW1.md')))
    P('No change in either draft: the same 12 (v2) / 13 (style) findings, all present at PW-1 final. The three "unsourced" lines are '
      'the `2027` of `iclr2027` in the preamble (`\\input@path`, the `\\RequirePackage` line) and the `\\bibliographystyle` line; the not-found tokens are '
      'strings their cited file does not print verbatim (the K-grid list, the App. E byte totals, and the seed-5 count 106, which moved with item 5 into '
      'Tab. cmazet-sym). Checked number tokens fell 2,597 → 2,342 because numbers left the prose in item 5.')
    P('')


def figtab_part():
    J = json.load(open(os.path.join(G, 'measure-after/FIGTAB-INDEX.json')))
    P('## 4. Figure / table index (number → label → caption lead → page → section)')
    P('')
    P('From the compiled `.aux` (`pw1_draft_final/fig_table_index.py`, unchanged): every label cross-checked against its list-of-figures/tables '
      'entry (caption start and page agree), and the label set equals the lof/lot set.')
    P('')
    for f, d in DR:
        x = J[d]; it = x['items']
        nf = len({v['num'] for v in it.values() if v['kind'] == 'figure'}); nt = len({v['num'] for v in it.values() if v['kind'] == 'table'})
        shared = [n for n, c in Counter((v['kind'], v['num']) for v in it.values()).items() if c > 1]
        bad = [k for k, v in it.items() if not v.get('lo_agrees')]
        P('### %s: %d figures, %d tables (%d labels; shared numbers: %s); lof/lot disagreements %d; lof/lot without label %s; label without lof/lot %s' % (
            d, nf, nt, len(it), ', '.join('%s %s carries %s' % (k, n, ' and '.join('`%s`' % l for l, v in it.items() if (v['kind'], v['num']) == (k, n))) for k, n in shared) or 'none',
            len(bad), x['lo_without_label'] or 'none', x['label_without_lo'] or 'none'))
        P('')
        P('| kind | number | label | caption lead | page | section |')
        P('|---|---|---|---|---|---|')
        for k, v in sorted(it.items(), key=lambda kv: (kv[1]['kind'], [int(z) if z.isdigit() else z for z in re.split(r'(\d+)', kv[1]['num'])])):
            P('| %s | %s | `%s` | %s | %d | %s %s |' % (v['kind'], v['num'], k, v['lead'].replace('|', '\\|')[:110], v['page'], v['section'], v['section_title'][:40]))
        P('')
    a = {k: (v['num'], v['page']) for k, v in J['DRAFT-v2']['items'].items()}
    b = {k: (v['num'], v['page']) for k, v in J['DRAFT-v2-style']['items'].items()}
    same_num = all(a[k][0] == b[k][0] for k in a) and set(a) == set(b)
    dp = [(k, a[k][1], b[k][1]) for k in a if a[k][1] != b[k][1]]
    P('Numbering identical between the drafts: **%s**. Pages differ for %d floats (the style twin is one page longer): %s.' % (
        same_num, len(dp), ', '.join('`%s` %d/%d' % x for x in dp)))
    P('')


def pages_part():
    P('## 5. Page counts per section (marker method C, `pw2_draft_sections/page_markers_c.py` + `pw1_draft_final/pages.py`)')
    P('')
    P('Exact heading positions at shipout (a `\\pdfsavepos` marker inside each heading). The marker builds reproduce every `\\newlabel` '
      'of the unmarked build and 0 toc/marker page mismatches, in both drafts. "len" = text-flow length in pages from the heading to the '
      'next heading of the same or a higher level (floats count where they land). **Main text** = from the top of page 1 (title, abstract) '
      'to the "AI use statement" heading, as in PW-1 (statements, references and appendices do not count toward ICLR\'s limit).')
    P('')
    rows = {}
    for f, d in DR:
        rows[d] = json.load(open(os.path.join(G, f'measure-after/PAGES-{d}.json')))
    def tab(d):
        J = rows[d]; R = J['rows']
        ai = next(r for r in R if 'AI use' in r['title'])
        intro = next(r for r in R if r['title'].startswith('Introduction'))
        ref = next(r for r in R if r['kind'] == 'bibliography')
        return J, R, ai, intro, ref
    P('| | DRAFT-v2 | DRAFT-v2-style |')
    P('|---|---|---|')
    Jv, Rv, aiv, iv, rv = tab('DRAFT-v2'); Js, Rs, ais, is_, rs = tab('DRAFT-v2-style')
    P('| total pages | %d | %d |' % (Jv['pages_total'], Js['pages_total']))
    P('| **main text (p. 1 top → AI use statement)** | **%.2f pp** (ends on p. %d) | **%.2f pp** (ends on p. %d) |' % (aiv['coord'], aiv['page'], ais['coord'], ais['page']))
    P('| main text against the 9-page limit | **+%.2f pp over** | **+%.2f pp over** |' % (aiv['coord'] - 9, ais['coord'] - 9))
    P('| of which title + abstract (p. 1 top → Introduction) | %.2f pp | %.2f pp |' % (iv['coord'], is_['coord']))
    P('| Introduction → AI use statement | %.2f pp | %.2f pp |' % (aiv['coord'] - iv['coord'], ais['coord'] - is_['coord']))
    P('')
    P('| # | heading | DRAFT-v2: page, len (pp) | DRAFT-v2-style: page, len (pp) |')
    P('|---|---|---|---|')
    for a, b in zip(Rv, Rs):
        assert a['title'] == b['title'], (a['title'], b['title'])
        if a['kind'] == 'end': continue
        t = a['title']
        if a['kind'] == 'section*' and t.startswith('Appendices'): t = 'Appendices (preface)'
        ind = '&nbsp;&nbsp;' if a['kind'].startswith('sub') and a['kind'] != 'subsubsection*' else ''
        if a['kind'] == 'subsubsection*': t = 'Statement: ' + t
        num = a['num'] or ''
        fl = lambda r: '%d, %.2f' % (r['page'], r['length']) if r['length'] is not None else '%d' % r['page']
        bold = a['kind'] in ('section', 'bibliography') or a['kind'] == 'subsubsection*'
        P('| %s | %s%s | %s | %s |' % (num, ind, ('**%s**' % t) if a['kind'] == 'section' and a['title'] in ('Introduction', 'Methods', 'Experiments', 'Results', 'Discussion') else t, fl(a), fl(b)))
    P('')
    main = ['Introduction', 'Methods', 'Experiments', 'Results', 'Discussion']
    P('Five main sections (len, pp): ' + '; '.join('%s %.2f / %.2f' % (m, next(r for r in Rv if r['title'] == m)['length'], next(r for r in Rs if r['title'] == m)['length']) for m in main) + ' (v2 / style). '
      'Statements (AI use + ethics + reproducibility): %.2f / %.2f. References: %.2f / %.2f. Appendices (preface to end): %.2f / %.2f.' % (
          rv['coord'] - aiv['coord'], rs['coord'] - ais['coord'], rv['length'], rs['length'],
          Rv[-1]['coord'] - next(r for r in Rv if r['title'].startswith('Appendices'))['coord'], Rs[-1]['coord'] - next(r for r in Rs if r['title'].startswith('Appendices'))['coord']))
    P('')


def numbers_part():
    P('## 6. Every number left in the main text, with its justification (re-derived independently)')
    P('')
    P('Scanner and classifier written fresh for this audit (`pw2_draft_final/numscan.py`, `numclass.py`); they import nothing from '
      '`pw2_draft_numbers`. Main text = `\\title` to the "AI use statement" heading, abstract included; comments, floats (captions and tables), '
      '`\\figslot` calls and `\\ref/\\label/\\cite` keys excluded. Tokens: every digit run, and every spelled cardinal (zero … twenty, '
      'the tens, hundred, thousand, million, billion, half, twice, dozen). Each token gets a class and a justification; **UNCLASSIFIED: 0 in '
      'both drafts.** Every rounded value is re-checked against its record, verbatim (`rounded_check.py`, 32 phrases, **0 FAIL** in both drafts; '
      'planted control FAILs). Full per-token lists: `NUMBERS-NOW-DRAFT-v2.md`, `NUMBERS-NOW-DRAFT-v2-style.md`.')
    P('')
    for f, d in DR:
        T = json.load(open(os.path.join(G, f'numbers/{d}-class.json')))
        secs = []
        for t in T:
            if t['section'] not in secs: secs.append(t['section'])
        cls = ['HEAD-HELD', 'IDENTIFIER', 'PROTOCOL/DEFINITION', 'ROUNDED', 'COUNT', 'NOT-A-NUMBER', 'UNCLASSIFIED']
        P('### %s: %d tokens (%d digit, %d word)' % (d, len(T), sum(t['kind'] == 'digit' for t in T), sum(t['kind'] == 'word' for t in T)))
        P('')
        P('| section | digit | word | ' + ' | '.join(cls) + ' |')
        P('|---|---|---|' + '---|' * len(cls))
        for s in secs:
            ts = [t for t in T if t['section'] == s]
            P('| %s | %d | %d | %s |' % (s, sum(t['kind'] == 'digit' for t in ts), sum(t['kind'] == 'word' for t in ts), ' | '.join(str(sum(t['cls'] == c for t in ts)) for c in cls)))
        P('| **total** | %d | %d | %s |' % (sum(t['kind'] == 'digit' for t in T), sum(t['kind'] == 'word' for t in T), ' | '.join(str(sum(t['cls'] == c for t in T)) for c in cls)))
        P('')
        hd = [t for t in T if t['cls'] == 'HEAD-HELD' and t['kind'] == 'digit']
        P('Digit tokens outside Head-held text: %d; inside Head-held text: %d (%s).' % (
            sum(t['kind'] == 'digit' and t['cls'] != 'HEAD-HELD' for t in T), len(hd),
            ', '.join('%s %d' % (k, v) for k, v in Counter(t['why'].split(' (')[0] for t in hd).items())))
        P('')
    for f, d in DR:
        T = json.load(open(os.path.join(G, f'numbers/{d}-class.json')))
        P('### Abstract, %s' % d)
        P('')
        for t in T:
            if t['section'] == 'Abstract': P('- l.%d `%s` — %s: %s' % (t['line'], t['tok'], t['cls'], t['why']))
        q = [t for t in T if t['section'] == 'Abstract' and t['cls'] in ('HEAD-HELD', 'ROUNDED', 'COUNT')]
        P('')
        P('**Quantities in the abstract: %d** (%s); the plan\'s limit is five. Names ("Shapes3D", "zero-dynamics", "zero-step") and the pronoun "one" are not quantities.' % (len(q), ', '.join('"%s"' % t['tok'] for t in q)))
        P('')
    T = json.load(open(os.path.join(G, 'numbers/DRAFT-v2-class.json')))
    P('### Every number outside Head-held text (DRAFT-v2), with its justification')
    P('')
    P('| line | section | number | class | justification | context |')
    P('|---|---|---|---|---|---|')
    for t in T:
        if t['cls'] == 'HEAD-HELD': continue
        P('| %d | %s | `%s` | %s | %s | `%s` |' % (t['line'], t['section'], t['tok'], t['cls'], t['why'].replace('|', '\\|'), ' '.join(t['code'][max(0, t['col'] - 35):t['col'] + 45].split()).replace('|', '\\|').replace('`', "'")))
    P('')
    P('The Head-held tokens (Experiments first paragraph: the per-K seed SDs, per-seed footprints and gaps; Results first paragraph: '
      'the per-seed margins, the mean +7.034, +3.73, wins/losses, parameter and step counts, baseline ranges) are listed with their lines '
      'in `NUMBERS-NOW-DRAFT-v2.md`. They are per-seed and table-readable values that item 5 would otherwise remove; PW-2 may not touch them.')
    P('')
    P('### Rounded values, re-checked against their records (DRAFT-v2; style identical, 32 rows, 0 FAIL)')
    P('')
    P(rd('numbers/ROUNDED-DRAFT-v2.md'))
    P('')
    P('### Disagreements with pw2-draft-numbers / pw2-draft-leftovers (`NUMBERS-LEFT.md`, the latest list)')
    P('')
    for f, d in DR:
        c = rd(f'numbers/COMPARE-{d}.md')
        P(c.replace('### ', '#### '))
        P('')
    P('**Substantive disagreement: one.** "more than **eighteen** tuples per held-out combination" (Results, Item memory; v2 l.1034, style l.1028) '
      'is a number in the main text that `NUMBERS-LEFT.md` does not list (its word list stops before "eighteen"). It is a rounded floor, '
      'and its record holds (3-seed mean 18.56 > 18; `w13-seeds-and-tables/TABLES.md`, rounded_check row 19). Every other token-level difference is '
      'the tokeniser: `NUMBERS-LEFT` skips a digit glued to letters (Shapes3D, S1–S4, float64, `$n\\ge3$`, `$K\\ge36$`, `$n=3\\to6$`) '
      'and splits comma lists differently. The class differences are taxonomy, not substance: this audit classes "zero" in "the interval lies above zero" '
      'as a reference point (not a count), "ten times / thirty times its steps" as the schedule name, "either half" and "half-life" as not numbers, '
      'and "about four thousand", "about three months", "four times the n-CLU\'s content rows" (and, in the style twin, "more than four of the six claim atoms") '
      'as rounded or exact values with a record (all re-checked) where `NUMBERS-LEFT` calls them count words. No number was found that item 5 should have moved and did not, outside Head-held text.')
    P('')


def tells_part():
    t = rd('tells/TELLS-NOW.md')
    P('## 7. Per-tell counts per section, now (pw2-draft-tells\' detectors, unchanged)')
    P('')
    P('Columns: pw2-draft-tells\' before → after (its `TELLS-COUNT.md`) → **now**. `[nh]` = hits in Head-held text. ◆ = now differs from after '
      '(changed by pw2-draft-leftovers). Main-text sections below; every appendix row is in `TELLS-NOW.md`.')
    P('')
    body = t.split('## Sections where')[0].split('\n')
    keep = ('| section', '|---', '| Abstract', '| Introduction', '| Methods', '| Experiments', '| Results', '| Discussion', '| Statements', '| **total**')
    for l in body:
        if l.startswith('## T'): P('#### ' + l[3:]); continue
        if l.startswith(keep): P(l)
        elif l == '' and L and L[-1].startswith('|'): P('')
    P('')
    P(t.split('## Sections where')[1].join(['#### Sections where', '']) if '## Sections where' in t else '')
    P('')
    P('Reading the ◆ rows: Discussion T4 12 → 1 is item 5 on the limitations list (pw2-draft-leftovers part 7; the one hit left is "256 requests", '
      'an assertion size). App R T2 1 → 0 and T7 0 → 1: pw2-draft-leftovers cut two sentences from App R (Placeholder index) after the placeholders '
      'were removed; the contrast went with them, and the paragraph lost its only short sentence. Positive controls: `pw2_draft_tells/controls.py` on the '
      'live v2 draft, ALL CONTROLS PASS (every detector CAUGHT, the brief\'s seven examples MATCHED).')
    P('')


def hh_part():
    P('## 8. Head-held paragraphs: byte-equal to the latest HEAD-HELD.md')
    P('')
    P('Latest list: `outputs/pw2-draft-leftovers/HEAD-HELD.md` (= pw2-draft-tells\' list, unchanged by item 4). Two independent checks per draft: '
      '`pw2_draft_sections/headheld.py` (sha256 per paragraph) and a raw-line byte comparison of the lines HEAD-HELD.md names in the draft it was written '
      'from (pw2-draft-tells\' install = pw2-draft-leftovers\' `snap-start/`) against the lines headheld.py locates now (`scans.py headheld`).')
    P('')
    for f, d in DR:
        P('### ' + d)
        P('')
        P(rd(f'checks/HEADHELD-RAW-{d}.md'))
        P('')
        P('headheld.py: ' + rd(f'checks/HEADHELD-{d}.txt').strip().split('\n')[-1])
        P('')


def todo_part():
    P('## 9. Every remaining `\\TODO` and `\\PLACEHOLDER`, by argument')
    P('')
    for f, d in DR:
        P('### ' + d)
        P('')
        P(rd(f'checks/TODO-{d}.md'))
        P('')
    P('`\\TODO` 3 and `\\PLACEHOLDER` 0 in each draft, with byte-identical arguments between the drafts: the abstract\'s maze sentence (Head-held), '
      'the AI use statement (the Head\'s) and the anonymised code archive. The preamble keeps both macro definitions and their convention comments '
      '(pw2-draft-leftovers\' Head note 3); App. R "Placeholder index" now indexes no placeholder (its Head note 2).')
    P('')


def cite_part():
    P('## 10. Citation-scope audit of `jawahar2026chlu`')
    P('')
    P('Every sentence that carries the key (code, comments stripped), its topics and verdict (`scans.py cite`). Allowed: the unit, its dynamics '
      '(Hamiltonian, symplectic, settling), the relativistic governor. Disallowed in a citing sentence: memory, deletion/forgetting/unlearning, '
      'test-time compute, retention, GMOR, dissipation. Exit 1 on VIOLATION; planted control (the key cited for "the retention law") → VIOLATION.')
    P('')
    for f, d in DR:
        P('### ' + d)
        P('')
        P(rd(f'checks/CITE-{d}.md'))
        P('')
    P('Three citations per draft, all for the unit and its dynamics; **0 VIOLATION**. The Methods sentence that names "a factorized memory unit" '
      'and "deletion" carries no citation and attributes both to "we". The bibliography entry renders as "Chlu: The causal hamiltonian learning unit …" '
      '(lower-cased by the .bst, reported by pw2-draft-methods; the bib is the Head\'s).')
    P('')


def bib_part():
    P('## 11. Bibliography')
    P('')
    P(rd('checks/BIB.txt'))
    P('')


P('# FINAL-INDEX — the closing audit of PW-2 (both drafts, as installed after pw2-draft-leftovers)')
P('')
P('Session `pw2-draft-final`, branch `pw2/draft-final`, batmac. Read-only audit of `~/clu/pw2/live/v2/DRAFT-v2.tex` and '
  '`~/clu/pw2/live/style/DRAFT-v2-style.tex`. **No content edit, no mechanical fix needed, nothing installed.** Working folder: `%s`.' % os.path.basename(G.rstrip('/')))
P('')
P(rd('checks/SUMMARY.md'))
P('')
for fn in (compile_part, style_part, prov_part, figtab_part, pages_part, numbers_part, tells_part, hh_part, todo_part, cite_part, bib_part):
    fn()
P('## Positive controls (every scan; planted copies under `controls/`, never the live drafts)')
P('')
P('```')
P(rd('controls/CONTROLS.txt').strip())
P('```')
open(os.path.join(OUT, 'FINAL-INDEX.md'), 'w').write('\n'.join(L) + '\n')
for f, d in DR:
    open(os.path.join(OUT, f'NUMBERS-NOW-{d}.md'), 'w').write('# NUMBERS-NOW — every number in the main text of %s, class and justification (this audit)\n\n' % d + rd(f'numbers/{d}-class.md'))
print('wrote', os.path.join(OUT, 'FINAL-INDEX.md'))
