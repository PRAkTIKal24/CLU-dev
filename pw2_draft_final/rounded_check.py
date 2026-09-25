"""Every rounded phrase left in the main text, re-checked against its record (written fresh for the closing audit).

Each row: an anchor regex that must match the draft's main text (code, comments stripped, whitespace collapsed),
the record file(s) (relative to the evidence root ~/clu/evidence/outputs/), the recorded values that must occur
VERBATIM in the record, and the rounding rule evaluated on those values. Exit 1 on any FAIL.
usage: python3 rounded_check.py DRAFT.tex [--plant]   (--plant adds a planted row whose rule must FAIL)
"""
import os, re, sys
EV = os.path.expanduser('~/clu/evidence/outputs/')

R = [
 # (anchor regex, files, verbatim values, rule(values as floats) -> bool, rule text)
 (r'produces about seven more distinct', ['w20-shapes-data-matched/SHAPES-MATCHED.md'], ['7.034'],
  lambda v: round(v[0]) == 7, 'mean paired margin over the MLP +7.034 rounds to 7'),
 (r'by about seven tuples per held-out factor combination over the strongest', ['w20-shapes-data-matched/SHAPES-MATCHED.md'], ['7.034'],
  lambda v: round(v[0]) == 7, 'Contribution 1: +7.034 rounds to 7'),
 (r'by about seven tuples per held-out combination over the strongest', ['w20-shapes-data-matched/SHAPES-MATCHED.md'], ['7.034'],
  lambda v: round(v[0]) == 7, 'Summary: +7.034 rounds to 7'),
 (r'survivors fall by about six tuples', ['w20-shapes-data-matched/FINDINGS.md'], ['6.13', '6.43', '6.30'],
  lambda v: all(round(x) == 6 for x in v), 'abstract: per-seed survivor losses 6.13 / 6.43 / 6.30 each round to 6'),
 (r'surviving combinations by about six tuples per held-out combination, on all three seeds', ['w20-shapes-data-matched/FINDINGS.md'], ['6.13', '6.43', '6.30'],
  lambda v: all(round(x) == 6 for x in v), 'Sec. 4.3: 6.13 / 6.43 / 6.30 each round to 6'),
 (r'its own loss on that deletion, about six tuples', ['w20-shapes-data-matched/FINDINGS.md'], ['6.13', '6.43', '6.30'],
  lambda v: all(round(x) == 6 for x in v), 'Sec. 4.4: 6.13 / 6.43 / 6.30 each round to 6'),
 (r'surviving combinations by about six tuples per held-out combination, about thirty', ['w20-shapes-data-matched/FINDINGS.md'], ['6.13', '6.43', '6.30', '32.1'],
  lambda v: all(round(x) == 6 for x in v[:3]) and round(v[3], -1) == 30, 'Limitation 2: 6.13 / 6.43 / 6.30 -> 6; 32.1 -> 30 (nearest ten)'),
 (r'by about thirty each', ['w20-shapes-data-matched/FINDINGS.md'], ['32.1'],
  lambda v: round(v[0], -1) == 30, 'Sec. 4.3: -32.1 at seed 0 rounds to 30 (nearest ten)'),
 (r'in another colour by about one', ['w20-shapes-data-matched/FINDINGS.md'], ['1.47'],
  lambda v: round(v[0]) == 1, 'Sec. 4.3: -1.47 rounds to 1'),
 (r'by at most about a tenth of a tuple', ['w20-shapes-data-matched/FINDINGS.md'], ['0.11'],
  lambda v: round(v[0], 1) == 0.1, 'largest item-deletion move 0.11 rounds to 0.1'),
 (r'screened about four thousand unique works', ['_thinkpad/w8/w8-novelty/NOVELTY.md'], ['4,050'],
  lambda v: round(v[0], -3) == 4000, 'literature screen 4,050 rounds to 4,000'),
 (r'valid on about nine in ten', ['../../outputs/pw1-render-shapes/REQUESTS-pw1.json'], ['5177', '5688'],
  lambda v: round(v[0] / v[1], 1) == 0.9, 'n-CLU valid 5,177 of 5,688 at seed 0 = 0.910 -> 0.9'),
 (r'moves the LSTM by about one tuple', ['w20-shapes-data-matched/FINDINGS.md'], ['20.55', '21.85'],
  lambda v: round(v[1] - v[0]) == 1, 'LSTM ten-times spot check 20.55 -> 21.85, +1.30 -> 1'),
 (r'up to about four novel tuples', ['w21-ladder-rivals-32k/FINDINGS.md'], ['4.13'],
  lambda v: round(v[0]) == 4, 'seed-to-seed MLP difference up to 4.13 -> 4'),
 (r'rise, nearly five tuples', ['w15-coverage-control/FINDINGS.md'], ['4.989'],
  lambda v: 4.5 <= v[0] < 5, 'TEST rise n=3 -> 6 +4.989: below five and within 10%'),
 (r'moves by about 0.6 relative RMS', ['w13-seeds-and-tables/TABLES.md'], ['0.610083', '0.614277'],
  lambda v: all(round(x, 1) == 0.6 for x in v), 'relative RMS 0.610083 (red) / 0.614277 (others) -> 0.6'),
 (r'just under 1.4 times the reference', ['w13-seeds-and-tables/TABLES.md'], ['712,072', '719,352', '524,288'],
  lambda v: 1.3 < v[1] / v[2] < 1.4 and v[0] / v[2] < 1.4, 'composition-only bytes 712,072..719,352 / 524,288 = 1.358..1.372 < 1.4'),
 (r'within half a percent', ['w13-seeds-and-tables/TABLES.md'], ['-0.304%', '+0.495%'],
  lambda v: max(abs(x) for x in v) < 0.5, 'byte/parameter match -0.304% .. +0.495%, all |.| < 0.5%'),
 (r'more than eighteen tuples', ['w13-seeds-and-tables/TABLES.md'], ['-21.0816', '-16.5306', '-18.0816'],
  lambda v: -sum(v) / 3 > 18, 'item memory row 4 minus learned reference, 3-seed mean of 21.08 / 16.53 / 18.08 = 18.56 > 18'),
 (r'at about four times the reference bytes', ['w15-lace-poe-rows/FINDINGS.md'], ['4.09', '4.11'],
  lambda v: all(round(x) == 4 for x in v), 'nearest-prototype ceiling at 4.09 .. 4.11 x the reference -> 4'),
 (r'accuracy of about nine in ten', ['w21-maze-t-seed4/FINDINGS.md', 'w21-rel-newseeds/FINDINGS.md'], ['0.900', '0.930'],
  lambda v: all(round(x, 1) == 0.9 for x in v), 'per-step accuracy 0.900 .. 0.930 -> 0.9'),
 (r'at \$\\Kstar\$ is about four times the reference', ['w13-referee-2/REFEREE-v1.md'], ['4.0897', '4.0768', '4.0847'],
  lambda v: all(round(x) == 4 for x in v), 'deletable object 4.0897 / 4.0768 / 4.0847 x the reference -> 4'),
 (r'still gains more than a tuple', ['w20-shapes-data-matched/FINDINGS.md'], ['1.47'],
  lambda v: v[0] > 1, 'MLP seed 0, ten -> thirty times its steps +1.47 > 1'),
 (r'by less than one six-factor tuple', ['w12-clue-theorist/CLUE.md'], ['0.531', '0.510', '0.388'],
  lambda v: all(x < 1 for x in v), 'gentlest dynamics setting lowers composition by 0.531 / 0.510 / 0.388 < 1'),
 (r'by about forty on the seed mean', ['w12-clue-theorist/CLUE.md'], ['39.959', '46.918', '39.061'],
  lambda v: round(sum(v) / 3, -1) == 40, 'full strength -39.959 / -46.918 / -39.061, seed mean 41.98 -> 40 (nearest ten)'),
 (r'nearly two thirds of the image differs', ['w10-physics-section-evidence/EVIDENCE.md'], ['65.17', '65.39'],
  lambda v: 60 < max(v) < 200 / 3, 'image differs 65.17% .. 65.39%: below 2/3 (66.67%) and within 10%'),
 (r'refit takes longer than a baseline', ['w20-shapes-data-matched/FINDINGS.md', 'w20-shapes-data-matched/SHAPES-MATCHED.md'], ['2,367', '2064.5'],
  lambda v: v[0] > v[1], 'fastest book refit 2,367 s > slowest baseline retrain 2,064.5 s'),
 (r'score far below one distinct valid novel tuple', ['w15-lace-poe-rows/FINDINGS.md'], ['0.117', '0.033', '0.267', '0.224', '0.122', '0.184', '0.020'],
  lambda v: max(v) < 1 / 3, 'LACE rows 3-4 per seed 0.117 / 0.033 / 0.267 and 0.224 / 0.122 / 0.184, PoE at most 0.020: all below a third of a tuple'),
 (r'(more than four of the six claim atoms|reached by any single work is four)', ['_thinkpad/w8/w8-novelty/NOVELTY.md'], ['4 of 6'],
  lambda v: True, 'an exact count: highest claim-atom count of any single work "4 of 6" (the record)'),
 (r'younger than about three months', ['_thinkpad/w8/w8-novelty/NOVELTY.md'], ['< ~3 months'],
  lambda v: True, 'the record itself states the approximate window "< ~3 months"'),
 (r'rows, four times the n-CLU', ['w21-ladder-rivals-32k/CAPTION.md'], ['131,072', '32,768'],
  lambda v: v[0] / v[1] == 4, 'an exact ratio: data-matched pool 131,072 rows / n-CLU content rows 32,768 = 4'),
 (r'retained statistic is nearly a gigabyte', ['w20-shapes-data-matched/SHAPES-MATCHED.json'], ['967304800'],
  lambda v: 0.9e9 <= v[0] < 1e9, '967,304,800 B = 0.967 GB (decimal): below 1 GB, within 10%'),
]
PLANT = (r'produces about seven more distinct', ['w20-shapes-data-matched/SHAPES-MATCHED.md'], ['7.034'],
         lambda v: round(v[0]) == 8, 'PLANTED: 7.034 claimed to round to 8 (must FAIL)')


def fnum(s):
    if not re.search(r'\d', s): return 0.0
    s = re.sub(r'^[^\d+-]*', '', s).split(' ')[0]
    return float(s.replace(',', '').replace('%', '').replace('+', ''))


def main():
    path = sys.argv[1]
    rows = R + ([PLANT] if '--plant' in sys.argv else [])
    L = open(path).read().split('\n')
    t0 = next(i for i, l in enumerate(L) if l.startswith('\\title{'))
    t1 = next(i for i, l in enumerate(L) if l.startswith('\\subsubsection*{AI use statement}'))
    code = ' '.join(re.sub(r'(?<!\\)%.*', '', l) for l in L[t0:t1])
    code = ' '.join(code.split())
    fails = 0
    print('| # | anchor (main text) | in text | record file(s) | values verbatim | rule | result |')
    print('|---|---|---|---|---|---|---|')
    for i, (a, files, vals, rule, txt) in enumerate(rows, 1):
        intext = len(re.findall(a, code))
        body = ''
        for f in files:
            try: body += open(os.path.join(EV, f), errors='replace').read() + '\n'
            except OSError: body += ''
        found = [v for v in vals if v in body]
        ok_found = len(found) == len(vals)
        try: ok_rule = ok_found and rule([fnum(v) for v in vals])
        except Exception: ok_rule = False
        res = 'PASS' if (intext >= 1 and ok_found and ok_rule) else 'FAIL'
        fails += res == 'FAIL'
        print('| %d | `%s` | %d | %s | %s | %s | **%s** |' % (i, a.replace('\\', ''), intext, '; '.join('`%s`' % f for f in files),
              ', '.join(('%s ✓' % v) if v in found else ('%s MISSING' % v) for v in vals), txt, res))
    print('\n%d rows, %d FAIL' % (len(rows), fails))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
